import argparse
import json
import os
import pickle
import time

from invpend_env import PendulumEnv
from mixed_integer_encoding import (
    ReluNetwork,
    MixedIntegerEncoding,
    clip_at_pm1,
    clip_at_pm_k,
)
import tensorflow as tf
import numpy as np
import mip
import datetime

from lds_env import LDSEnv
import matplotlib.pyplot as plt
import seaborn as sns

VERBOSITY = 1

tf.random.set_seed(1234)
default_rng = np.random.default_rng(98234)


def encode_relu(opt_model, x, lb, ub):
    binary_var = opt_model.add_var(
        var_type=mip.BINARY,
        # lowlbBound=0,
        # upBound=1,
        name="{}_binary".format(x.name),
    )
    # y >= 0
    post_act_neuron = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=0,
        ub=np.max([ub, 0]),
        name="{}_relu".format(x.name),
    )

    # y >= x
    opt_model += post_act_neuron - x >= 0

    # y <= ub*d
    opt_model += ub * binary_var - post_act_neuron >= 0

    # y <= x - lb*(1-d)
    opt_model += x + binary_var * lb + post_act_neuron * -1 >= lb
    return post_act_neuron


def encode_angular(opt_model, x, lb, ub):
    xs1 = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=lb - np.pi / 2,
        ub=ub + np.pi / 2,
        name="s_x1",
    )
    opt_model += xs1 == x - np.pi / 2
    xs2 = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=lb - 3 * np.pi / 2,
        ub=ub + 3 * np.pi / 2,
        name="s_x2",
    )
    opt_model += xs2 == x - 3 * np.pi / 2
    ys1 = encode_relu(opt_model, xs1, xs1.lb, xs1.ub)
    ys2 = encode_relu(opt_model, xs2, xs2.lb, xs2.ub)

    y = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=-1,
        ub=1,
        name="angular",
    )
    opt_model += (
        x * (1 / (3.1415 / 2)) + -2 / (3.1415 / 2) * ys1 + 2 / (3.1415 / 2) * ys2 == y
    )
    return y


def encode_pend(opt_model, state, action):
    # State transition of the inv-pend env
    th, thdot = state
    max_speed = 8
    max_torque = 2.5
    dt = 0.05
    g = 10.0
    m = 0.8  # 1.0
    l = 1.0

    sin_th = encode_angular(opt_model, th + np.pi, th.lb + np.pi, th.ub + np.pi)

    newthdot = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=thdot.lb
        + (-3 * g / (2 * l) * 1 + 3.0 / (m * l ** 2) * max_torque * action.lb) * dt,
        ub=thdot.ub
        + (-3 * g / (2 * l) * (-1) + 3.0 / (m * l ** 2) * max_torque * action.ub) * dt,
        name="new_thdot",
    )
    opt_model += (
        newthdot
        == thdot
        + (-3 * g / (2 * l) * sin_th + 3.0 / (m * l ** 2) * max_torque * action) * dt
    )
    newthdot = clip_at_pm_k(opt_model, newthdot, max_speed)
    newth = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=th.lb + newthdot.lb * dt,
        ub=th.ub + newthdot.ub * dt,
        name="new_th",
    )
    opt_model += newth == th + newthdot * dt
    return (newth, newthdot)


def encode_lds(opt_model, state, action):
    x, y = state
    # new_y = y + action * 0.2
    # new_x = x + new_y * 0.3 + action * 0.05
    new_y = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=y.lb + 0.2 * action.lb,
        ub=y.ub + 0.2 * action.ub,
        name="new_y",
    )
    opt_model += y + 0.2 * action == new_y
    new_x = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=x.lb + 0.3 * new_y.lb + 0.05 * action.lb,
        ub=x.ub + 0.3 * new_y.ub + 0.05 * action.ub,
        name="new_x",
    )
    opt_model += x + 0.3 * new_y + 0.05 * action == new_x
    return (new_x, new_y)


def build_dataset(x, y, s, n, batch_size=64, ce_bs=4):
    train_ds = tf.data.Dataset.from_tensor_slices((x, y))
    ce_ds = tf.data.Dataset.from_tensor_slices((s, n)).repeat()
    fake_labels = tf.data.Dataset.from_tensor_slices(tf.ones(10)).repeat()

    train_ds = train_ds.shuffle(1000)
    ce_ds = ce_ds.shuffle(100)

    train_ds = train_ds.batch(batch_size)
    ce_ds = ce_ds.batch(ce_bs)

    full_ds = tf.data.Dataset.zip((train_ds, ce_ds))
    # Need to add a fake label because of TensorFlow keras.fit
    fake_lablled_full_ds = tf.data.Dataset.zip((full_ds, fake_labels)).prefetch(
        tf.data.AUTOTUNE
    )
    return fake_lablled_full_ds


class CounterExampleBuffer:
    def __init__(self):
        self.s = []
        self.n = []

    def append(self, s, n):
        self.s.append(s)
        self.n.append(n)

    def __len__(self):
        return len(self.s)

    def materialize(self):
        return np.stack(self.s, axis=0), np.stack(self.n, axis=0)


class PairTrainer(tf.keras.Model):
    def __init__(self, relunet, eps):
        super(PairTrainer, self).__init__()
        self.eps = eps
        self.relunet = relunet
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.accuracy_fn = tf.keras.metrics.BinaryAccuracy(
            name="accuracy", threshold=0.0
        )

    def call(self, inputs, **kwargs):
        (state, label), (us_state, us_next_state) = inputs
        ind_s = self.relunet(state)
        ind_us_s = self.relunet(us_state)
        ind_us_next = self.relunet(us_next_state)

        cls_loss = self.loss_fn(label, ind_s)
        self.add_loss(tf.reduce_mean(cls_loss))
        self.add_metric(self.accuracy_fn(label, ind_s), name="accuracy")
        self.add_metric(cls_loss, name="cls_loss")

        us_pos_mask = tf.greater_equal(ind_us_s, -self.eps)
        next_neg_mask = tf.less_equal(ind_us_next, self.eps)
        aux_mask = tf.reshape(
            tf.cast(tf.logical_and(us_pos_mask, next_neg_mask), tf.float32), (-1,)
        )

        us_batch_size = tf.shape(us_state)[0]
        us_loss = self.loss_fn(tf.zeros((us_batch_size, 1)), ind_us_s)
        next_loss = self.loss_fn(tf.ones((us_batch_size, 1)), ind_us_next)
        aux_loss = aux_mask * (us_loss + next_loss)
        # print("aux_loss:", aux_loss)
        self.add_loss(tf.reduce_mean(aux_loss) * 0.2)  # Factor
        self.add_metric(aux_loss, name="aux_loss")


def log(msg):
    if VERBOSITY > 0:
        print(msg)


class VerificationLoop:
    def __init__(
        self,
        env,
        policy,
        ind_hidden_dim,
        train_eps,
        init_samples,
        gen_plot=False,
        bootstrap=False,
    ):
        os.makedirs("loop", exist_ok=True)
        self.ce_buffer = CounterExampleBuffer()
        self.env = env
        self.policy = policy
        self.bayes_eps = 0.0

        self.ind = ReluNetwork([ind_hidden_dim, 1])
        self.ind.compile(
            optimizer=tf.keras.optimizers.Adam(0.0005),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)],
        )

        self.twin = PairTrainer(self.ind, eps=train_eps)
        self.twin.compile(optimizer=tf.keras.optimizers.Adam(0.0005), run_eagerly=False)
        self.train_x, self.train_y = get_indicator_training_data(env, policy, bootstrap)
        log(f"Training indicator net on x={self.train_x.shape}, y={self.train_y.shape}")
        hist = self.ind.fit(self.train_x, self.train_y, batch_size=64, epochs=20)
        log(f"Indicator has accuracy {100 * hist.history['binary_accuracy'][-1]:0.2f}%")
        if init_samples > 0:
            self.initialize_ce_buffer(policy, N=init_samples)
            self.fine_tune()

        # Just for plotting
        self.gen_plot = gen_plot
        self._plot_trajectories = []
        for i in range(30):
            t = single_trajectory(policy, env, prefer_border=True)
            self._plot_trajectories.append(np.stack(t, axis=0))
        if self.gen_plot:
            self.plot_indicator("loop/step_0000.png")

    def eval_init_accuracy(self):
        loss, acc = self.ind.evaluate(self.train_x, self.train_y)
        return acc

    def initialize_ce_buffer(self, policy, N=50):
        while len(self.ce_buffer) < N:
            s = default_rng.uniform(-1, 1, size=(2,))
            if -0.5 < s[0] < 0.5 and -0.5 < s[1] < 0.5:
                continue
            n = single_step(policy, self.env, s)
            self.ce_buffer.append(s, n)

    def run_loop(self, max_steps=None, timeout=None):
        start_time = time.time()
        step = 0
        while max_steps is None or step < max_steps:
            step += 1
            print(f"Running MIP step {step}")
            ce = self.check_mip()
            if ce is None:
                # Done # Plot before exit
                if self.gen_plot:
                    self.plot_indicator(f"loop/step_{step:04d}.png")
                init_ok = self.check_init_mip()
                unsafe_ok = self.check_unsafe_mip()
                if not init_ok:
                    print(
                        "WARNING: There seems to be unsafe parts in the initial states"
                    )
                print(f"init_ok={init_ok}")
                if not unsafe_ok:
                    print("WARNING: There seems to be safe parts in the unsafe states")
                print(f"unsafe_ok={unsafe_ok}")
                if init_ok and unsafe_ok:
                    return True  # other conditions empirically checked (via accuracy on the samples of the two sets)
                return True  # other conditions empirically checked (via accuracy on the samples of the two sets)
            # if self.gen_plot:
            if step % 100 == 1 and self.gen_plot:
                self.plot_indicator(
                    f"loop/step_{step:04d}.png", (ce["s_pre"], ce["s_post"])
                )
            if ce is not None:
                batched_ce = np.stack((ce["s_pre"], ce["s_post"]), axis=0)
                f_check = self.ind.predict(batched_ce)
                a_check = self.policy.predict(batched_ce)[0, 0]
                # print(f"Ind pred BEFORE finetuning: ", f_check)
                # print(
                #     f"Sanity check action: tf {a_check:0.4g} vs mip {ce['action']:0.4g}"
                # )
                # print(
                #     f"Sanity check s_pre : tf {f_check[0,0]:0.4g} vs mip {ce['f_pre']:0.4g}"
                # )
                # print(
                #     f"Sanity check s_post: tf {f_check[1,0]:0.4g} vs mip {ce['f_post']:0.4g}"
                # )
                if ce["f_pre"] < -1e-4 or ce["f_post"] > 1e-4:
                    # If f(pre) is really negative or f(post) is really positive the MIP is wrong
                    raise ValueError("MIP is WRONG!")
                self.ce_buffer.append(ce["s_pre"], ce["s_post"])
            print(f"Finetuning step {step} with eps={self.bayes_eps:0.3g}")
            self.fine_tune()
            if ce is not None:
                print(f"Ind pred AFTER  finetuning: ", self.ind.predict(batched_ce))
            elapsed = time.time() - start_time
            if time is not None and elapsed > timeout:
                print("Time out reached")
                return False

    def fine_tune(self):
        s, n = self.ce_buffer.materialize()
        full_ds = build_dataset(self.train_x, self.train_y, s, n)
        self.twin.fit(full_ds, epochs=3)

    def get_state_bounds(self):
        if isinstance(self.env, LDSEnv):
            state_bound_x = 1.2
            state_bound_y = 1.2
        elif isinstance(self.env, PendulumEnv):
            state_bound_x = 2.8
            state_bound_y = 4
        else:
            raise ValueError("Something isn't right")
        return state_bound_x, state_bound_y

    def check_init_mip(self):

        opt_model = mip.Model()
        mip_ind = MixedIntegerEncoding(
            self.ind,
            [self.env.init_space.low[0], self.env.init_space.low[1]],
            [self.env.init_space.high[0], self.env.init_space.high[1]],
        )
        mip_ind.encode_network(opt_model=opt_model, name="ind")

        opt_model += mip_ind.output_vars[0] <= 0
        opt_model += mip_ind.input_vars[0] >= self.env.init_space.low[0]
        opt_model += mip_ind.input_vars[0] <= self.env.init_space.high[0]
        opt_model += mip_ind.input_vars[1] >= self.env.init_space.low[1]
        opt_model += mip_ind.input_vars[1] <= self.env.init_space.high[1]

        status = opt_model.optimize()
        if status == mip.OptimizationStatus.OPTIMAL:
            s_init = np.array([mip_ind.input_vars[0].x, mip_ind.input_vars[1].x])
            print("Init is <= 0")
            print("s_init=", str(s_init))
            s_check = self.ind.predict(np.expand_dims(s_init, 0))[0]
            print("s_check=", s_check)
            return False
        elif status == mip.OptimizationStatus.INFEASIBLE:
            return True
        else:
            raise ValueError("Unknown status: ", status)

    def check_unsafe_mip(self):

        all_unsat = True
        state_bound_x, state_bound_y = self.get_state_bounds()
        opt_model = mip.Model()
        mip_ind = MixedIntegerEncoding(
            self.ind, [-state_bound_x, -state_bound_y], [state_bound_x, state_bound_y]
        )
        mip_ind.encode_network(opt_model=opt_model, name="ind")

        opt_model += mip_ind.output_vars[0] >= 0
        opt_model += mip_ind.input_vars[0] >= state_bound_x

        status = opt_model.optimize()
        if status == mip.OptimizationStatus.OPTIMAL:
            all_unsat = False

        opt_model = mip.Model()
        mip_ind = MixedIntegerEncoding(
            self.ind, [-state_bound_x, -state_bound_y], [state_bound_x, state_bound_y]
        )
        mip_ind.encode_network(opt_model=opt_model, name="ind")

        opt_model += mip_ind.output_vars[0] >= 0
        opt_model += mip_ind.input_vars[0] <= -state_bound_x

        status = opt_model.optimize()
        if status == mip.OptimizationStatus.OPTIMAL:
            all_unsat = False

        return all_unsat

    def check_mip(self):

        state_bound_x, state_bound_y = self.get_state_bounds()
        opt_model = mip.Model()
        mip_ind1 = MixedIntegerEncoding(
            self.ind, [-state_bound_x, -state_bound_y], [state_bound_x, state_bound_y]
        )
        mip_ind1.encode_network(opt_model=opt_model, name="ind_pre")

        mip_policy = MixedIntegerEncoding(
            self.policy,
            [-state_bound_x, -state_bound_y],
            [state_bound_x, state_bound_y],
        )
        mip_policy.encode_network(
            opt_model=opt_model,
            input_vars=mip_ind1.input_vars,
            name="policy",
            bayes_eps=self.bayes_eps,
        )

        clipped_output = clip_at_pm1(opt_model, mip_policy.output_vars[0])

        if isinstance(self.env, LDSEnv):
            new_x, new_y = encode_lds(
                mip_policy.opt_model, mip_policy.input_vars, clipped_output
            )
        elif isinstance(self.env, PendulumEnv):
            new_x, new_y = encode_pend(
                mip_policy.opt_model, mip_policy.input_vars, clipped_output
            )
        else:
            raise ValueError("Hol'up something ain't right here")

        mip_ind2 = MixedIntegerEncoding(
            self.ind, [-state_bound_x, -state_bound_y], [state_bound_x, state_bound_y]
        )
        mip_ind2.encode_network(
            opt_model=mip_ind1.opt_model, input_vars=[new_x, new_y], name="ind_post"
        )
        mip_delta = 0
        opt_model += mip_ind1.output_vars[0] >= mip_delta  # 0
        opt_model += mip_ind2.output_vars[0] <= -mip_delta  # 0

        opt_model.objective = mip.maximize(
            mip_ind1.output_vars[0] - mip_ind2.output_vars[0]
        )  # Maximize delta

        print("Starting MIP at ", datetime.datetime.now().strftime("%H:%M:%S"))
        status = opt_model.optimize()
        print("Status: ", status)
        # print("opt_value {:0.2f}".format(opt_model.objective_value))

        if status == mip.OptimizationStatus.OPTIMAL:
            print("Counterexample found")
            s_pre = np.array([mip_ind1.input_vars[0].x, mip_ind1.input_vars[1].x])
            s_post = np.array([mip_ind2.input_vars[0].x, mip_ind2.input_vars[1].x])
            f_pre = mip_ind1.output_vars[0].x
            f_post = mip_ind2.output_vars[0].x
            action = mip_policy.output_vars[0].x

            return {
                "s_pre": s_pre,
                "s_post": s_post,
                "f_pre": f_pre,
                "f_post": f_post,
                "action": action,
            }
        elif status == mip.OptimizationStatus.INFEASIBLE:
            print("UNSAT!!!!!!!!")
            return None
        else:
            raise ValueError("Unknown status: ", status)

    def plot_indicator(self, filename="indicator.png", ce=None):

        state_bound_x, state_bound_y = self.get_state_bounds()

        X = np.linspace(-state_bound_x, state_bound_x, 100)
        Y = np.linspace(-state_bound_y, state_bound_y, 100)
        X, Y = np.meshgrid(X, Y)
        x = np.stack([X.flatten(), Y.flatten()], 1)
        pred = self.ind.predict(x, verbose=0)[:, 0]
        pos_xy = x[pred > 0]
        neg_xy = x[pred < 0]
        sns.set()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            pos_xy[:, 0],
            pos_xy[:, 1],
            color=sns.color_palette()[0],
            label="Pos",
            alpha=0.5,
        )
        ax.scatter(
            neg_xy[:, 0],
            neg_xy[:, 1],
            color=sns.color_palette()[1],
            label="Neg",
            alpha=0.5,
        )
        ax.scatter(
            [s[0] for s in self.ce_buffer.s],
            [s[1] for s in self.ce_buffer.s],
            color="red",
            alpha=0.7,
            marker="x",
        )
        ax.scatter(
            [s[0] for s in self.ce_buffer.n],
            [s[1] for s in self.ce_buffer.n],
            color="red",
            alpha=0.7,
            marker="x",
        )
        ax.scatter(
            self.train_x[self.train_y == 0, 0],
            self.train_x[self.train_y == 0, 1],
            color="tab:purple",
            alpha=0.7,
            zorder=10,
            marker="o",
        )
        ax.scatter(
            self.train_x[self.train_y == 1, 0],
            self.train_x[self.train_y == 1, 1],
            color="tab:olive",
            alpha=0.7,
            zorder=10,
            marker="o",
        )
        x_low = self.env.init_space.low[0]
        y_low = self.env.init_space.low[1]
        ax.plot(
            [x_low, -x_low, -x_low, x_low, x_low],
            [y_low, y_low, -y_low, -y_low, y_low],
            color="black",
            alpha=0.7,
            label="Init states",
        )
        for t in self._plot_trajectories:
            ax.plot(t[:, 0], t[:, 1], color="green", lw=0.8, alpha=0.9)

        p_dict = {
            "pos": (pos_xy[:, 0], pos_xy[:, 1]),
            "neg": (neg_xy[:, 0], neg_xy[:, 1]),
            "ce_s": self.ce_buffer.s,
            "ce_n": self.ce_buffer.n,
        }
        if ce is not None:
            s_pre, s_post = ce
            p_dict["ce"] = (s_pre[0], s_pre[1], s_post[0], s_post[1])
            ax.scatter([s_pre[0]], [s_pre[1]], color="red")
            ax.scatter([s_post[0]], [s_post[1]], color="red")
            ax.quiver(
                [s_pre[0]],
                [s_pre[1]],
                [s_post[0] - s_pre[0]],
                [s_post[1] - s_pre[1]],
                color=sns.color_palette()[0],
                label="Step that leaves 'safe' region",
            )

        ax.legend(loc="upper right").set_zorder(12)
        ax.set_title("Indicator image")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_ylim([-state_bound_y, state_bound_y])
        ax.set_xlim([-state_bound_x, state_bound_x])
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)


def policy_step(policy, x):
    action = policy.predict(np.expand_dims(x, 0))[0]
    return action


def single_trajectory(policy, env, prefer_border=False):
    done = False
    if prefer_border:
        s = env.observation_space.sample()
        s = np.clip(s, env.init_space.low, env.init_space.high)
        x = env.reset(s)
    else:
        x = env.reset()
    buffer = []
    while not done:
        buffer.append(x)
        action = policy_step(policy, x)
        x, r, done, info = env.step(action)
    return buffer


def single_step(policy, env, state):
    x = env.reset(state)
    action = policy_step(policy, x)
    next_state, _, _, _ = env.step(action)
    return next_state


def collect_trajectories(policy, env, n=100, prefer_border=False):
    buffer = []
    for i in range(n):
        buffer.extend(single_trajectory(policy, env, prefer_border))
    return np.stack(buffer, 0)


def sample_init_border(env, n):
    x1 = default_rng.uniform(env.init_space.low[0], env.init_space.high[0], n // 2)
    y1 = default_rng.choice([env.init_space.low[1], env.init_space.high[1]], n // 2)
    x2 = default_rng.choice([env.init_space.low[0], env.init_space.high[0]], n // 2)
    y2 = default_rng.uniform(env.init_space.low[1], env.init_space.high[1], n // 2)
    return np.concatenate([np.stack([x1, y1], 1), np.stack([x2, y2], 1)], 0)


def sample_unsafe_region_lds(env, policy, bootstrap, n=500):
    y1 = default_rng.uniform(-1.2, 1.2, n // 2)
    x1 = default_rng.choice([-1.2, 1.2], n // 2)
    x2 = default_rng.uniform(-1.2, 1.2, n // 2)
    y2 = default_rng.choice([-1.2, 1.2], n // 2)
    unsafe = np.concatenate([np.stack([x1, y1], 1), np.stack([x2, y2], 1)], 0)
    if bootstrap:
        print("Collecting unsafe data")
        buffer = []
        while len(buffer) < n:
            x = default_rng.uniform(-1.2, 1.2)
            y = default_rng.uniform(-1.2, 1.2)
            state = np.array([x, y])
            if not is_stable(env, policy, state):
                print(f"{len(buffer)}/{n}")
                buffer.append(state)
        bootstrapped = np.stack(buffer, axis=0)
        unsafe = np.concatenate([unsafe, bootstrapped], 0)
    return unsafe


def is_stable(env, policy, state):
    done = False
    x = env.reset(state)
    step = 0
    while not done:
        action = policy_step(policy, x)
        x, r, done, info = env.step(action)

        step += 1

    return step >= 200


def sample_unsafe_region_pend(env, policy, bootstrap, n=500):
    th1 = default_rng.uniform(-1, 1, n // 2)
    thdot1 = default_rng.choice([-2, 2], n // 2)
    th2 = default_rng.choice([0.9, -0.9], n // 2)
    thdot2 = default_rng.uniform(-2, 2, n // 2)
    unsafe = np.concatenate([np.stack([th1, thdot1], 1), np.stack([th2, thdot2], 1)], 0)
    if bootstrap:
        print("Collecting unsafe data")
        buffer = []
        while len(buffer) < n:
            th = default_rng.uniform(-np.pi / 2, np.pi / 2)
            thdot = default_rng.uniform(-3, 3)
            state = np.array([th, thdot])
            if not is_stable(env, policy, state):
                print(f"{len(buffer)}/{n}")
                buffer.append(state)
        buffer = np.stack(buffer, axis=0)
        unsafe = np.concatenate([unsafe, buffer], 0)
    return unsafe


def get_indicator_training_data(
    env,
    policy,
    bootstrap,
):
    pos_samples = collect_trajectories(policy, env, n=30, prefer_border=True)
    pos_samples = np.concatenate([pos_samples, sample_init_border(env, n=200)], 0)
    amount_of_unsafe = int(0.3 * pos_samples.shape[0])
    if isinstance(env, LDSEnv):
        neg_samples = sample_unsafe_region_lds(env, policy, bootstrap, amount_of_unsafe)
    else:
        neg_samples = sample_unsafe_region_pend(
            env, policy, bootstrap, amount_of_unsafe
        )

    pos_labels = np.ones(pos_samples.shape[0], dtype=np.int32)
    neg_labels = np.zeros(neg_samples.shape[0], dtype=np.int32)

    x = np.concatenate([pos_samples, neg_samples], 0)
    y = np.concatenate([pos_labels, neg_labels], 0)
    return x, y


def main():
    # run_study()
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", default=12, type=int)
    parser.add_argument("--env", default="lds")
    parser.add_argument("--policy", default="")
    parser.add_argument("--first_layer", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--single_step", action="store_true")
    parser.add_argument("--check_iu_mip", action="store_true")

    # env = gym.make("CartPole-v1")
    args = parser.parse_args()
    if args.env == "lds":
        env = LDSEnv()
    elif args.env == "pend":
        env = PendulumEnv()
    else:
        raise ValueError("Unknown env. Use either 'lds' or 'pend'")

    if args.first_layer:
        policy = ReluNetwork([4, 16, 1])
        policy(np.zeros((1, 2)))  # force weight creation
        policy.load_weights(f"weights/{args.env}_all.h5")
    else:
        policy = ReluNetwork([16, 1])
        policy(np.zeros((1, 2)))  # force weight creation
        policy.load_weights(f"weights/{args.env}.h5")

    loop = VerificationLoop(
        env,
        policy,
        ind_hidden_dim=args.hidden,
        init_samples=0,
        train_eps=0.02,
        gen_plot=True,
        bootstrap=args.bootstrap,
    )
    # for i, bayes_eps in enumerate([1e-10]):
    runtimes = []
    eps_solved = -1
    epss = []
    set_accuracy = []
    for i, bayes_eps in enumerate(
        [
            0,
            1e-10,
            1e-3,
            1e-2,
            2.5e-2,
            5e-2,
            7.5e-2,
            1e-1,
            1.25e-1,
            1.5e-1,
            1.75e-1,
            2e-1,
            5e-1,
            1.0,
            10,
        ]
    ):
        loop.bayes_eps = bayes_eps
        start_time = time.time()
        success = loop.run_loop(
            timeout=20 * 60, max_steps=1 if args.single_step else None
        )
        if success:
            runtimes.append(time.time() - start_time)
            epss.append(bayes_eps)
            # Evaluate accuracy of ind on initial set as empirical alternative to
            # MIP solving of the two constraints over the init and unsafe states
            set_accuracy.append(loop.eval_init_accuracy())
            eps_solved = bayes_eps
        text = "SUCCESS!" if success else "FAIL"
        print(
            f"Verification loop [{i}] of {args.env} with eps={bayes_eps:0.2g} results={text}"
        )

        if not success:
            break
    print(f"Eps solved: {eps_solved:0.2g}")
    result_dict = {
        "bayes_eps": epss,
        "runtimes": runtimes,
        "set_accuracy": set_accuracy,
        "args": vars(args),
        "last_time": time.time() - start_time,
    }


if __name__ == "__main__":
    main()