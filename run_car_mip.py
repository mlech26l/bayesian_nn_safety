import argparse
import json
import os
import time

from car_env import CarEnv
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

import matplotlib.pyplot as plt
import seaborn as sns

VERBOSITY = 1

tf.random.set_seed(1234)
default_rng = np.random.default_rng(98234)


def load(filename):
    arr = np.load(filename)
    return {k: arr[k] for k in arr.files}


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
        first_layer_trick=False,
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
        self.train_x, self.train_y = get_indicator_training_data(policy, env)

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
            t = single_trajectory(policy, env)
            self._plot_trajectories.append(np.stack(t, axis=0))
        if self.gen_plot:
            self.plot_indicator("loop/step_0000.png", train=True)

    def initialize_ce_buffer(self, policy, N=50):
        while len(self.ce_buffer) < N:
            px = default_rng.uniform(-self.env.bound_x, self.env.bound_x)
            ax = default_rng.uniform(-self.env.bound_x, self.env.bound_x)
            ay = default_rng.uniform(0, self.env.bound_y - 1)

            s = np.array([px, ax, ay])
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
                return True
            if step % 100 == 1 and self.gen_plot:
                self.plot_indicator(
                    f"loop/step_{step:04d}.png", (ce["s_pre"], ce["s_post"])
                )
            batched_ce = np.stack((ce["s_pre"], ce["s_post"]), axis=0)
            f_check = self.ind.predict(batched_ce)
            a_check = self.policy.predict(batched_ce)[0, 0]
            print(f"Ind pred BEFORE finetuning: ", f_check)
            print(f"Sanity check action: tf {a_check:0.4g} vs mip {ce['action']:0.4g}")
            print(
                f"Sanity check s_pre : tf {f_check[0,0]:0.4g} vs mip {ce['f_pre']:0.4g}"
            )
            print(
                f"Sanity check s_post: tf {f_check[1,0]:0.4g} vs mip {ce['f_post']:0.4g}"
            )
            # breakpoint()
            if ce["f_pre"] < -1e-4 or ce["f_post"] > 1e-4:
                # If f(pre) is really negative or f(post) is really positive the MIP is wrong
                raise ValueError("MIP is WRONG!")
            self.ce_buffer.append(ce["s_pre"], ce["s_post"])
            print(f"Finetuning step {step} with eps={self.bayes_eps:0.3g}")
            self.fine_tune()
            print(f"Ind pred AFTER  finetuning: ", self.ind.predict(batched_ce))
            elapsed = time.time() - start_time
            if time is not None and elapsed > timeout:
                print("Time out reached")
                return False

    def eval_init_accuracy(self):
        loss, acc = self.ind.evaluate(self.train_x, self.train_y)
        return acc

    def fine_tune(self):
        s, n = self.ce_buffer.materialize()
        full_ds = build_dataset(self.train_x, self.train_y, s, n)
        self.twin.fit(full_ds, epochs=3)

    def check_mip(self):

        # sin and cos have bound [-1,1]
        opt_model = mip.Model()
        px = opt_model.add_var(
            "px", lb=-self.env.bound_x, ub=self.env.bound_x, var_type=mip.INTEGER
        )
        ax = opt_model.add_var(
            "ax", lb=-self.env.bound_x, ub=self.env.bound_x, var_type=mip.INTEGER
        )
        ay = opt_model.add_var("ay", lb=0, ub=self.env.bound_y, var_type=mip.INTEGER)

        input_vars = [px, ax, ay]
        mip_ind1 = MixedIntegerEncoding(
            self.ind,
            [-self.env.bound_x, -self.env.bound_x, -1],
            [self.env.bound_x, self.env.bound_x, self.env.bound_y],
        )
        mip_ind1.encode_network(
            opt_model=opt_model, input_vars=input_vars, name="ind_pre"
        )

        mip_policy = MixedIntegerEncoding(
            self.policy,
            [-self.env.bound_x, -self.env.bound_x, -1],
            [self.env.bound_x, self.env.bound_x, self.env.bound_y],
        )

        mip_policy.encode_network(
            opt_model=opt_model,
            input_vars=input_vars,
            name="policy",
            bayes_eps=self.bayes_eps,
        )

        _, max_index = mip_policy.add_output_max_pooling([0, 1, 2], "max_out")

        ### ENV encoding
        inc_px = opt_model.add_var(
            "inc_px",
            lb=-self.env.bound_x - 1,
            ub=self.env.bound_x + 1,
            var_type=mip.INTEGER,
        )
        opt_model += inc_px == px + max_index[1] - max_index[2]
        new_px = clip_at_pm_k(opt_model, inc_px, 2)
        new_ay = opt_model.add_var(
            "new_ay",
            lb=-1,
            ub=self.env.bound_y,
            var_type=mip.INTEGER,
        )
        opt_model += new_ay == ay - 1
        new_ax = ax
        ### end of ENV encoding

        mip_ind2 = MixedIntegerEncoding(
            self.ind,
            [-self.env.bound_x, -self.env.bound_x, -1],
            [self.env.bound_x, self.env.bound_x, self.env.bound_y],
        )
        new_input_vars = [new_px, new_ax, new_ay]
        mip_ind2.encode_network(
            opt_model=mip_ind1.opt_model, input_vars=new_input_vars, name="ind_post"
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
            s_pre = np.array([v.x for v in mip_ind1.input_vars])
            s_post = np.array([v.x for v in mip_ind2.input_vars])
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

    def plot_indicator(self, filename="indicator.png", ce=None, train=False):

        X = np.linspace(-self.env.bound_x, self.env.bound_x, 25)
        Y = np.linspace(-1, self.env.bound_y, 25)
        X, Y = np.meshgrid(X, Y)
        px = np.zeros(Y.flatten().shape[0])
        x = np.stack([px, X.flatten(), Y.flatten()], 1)
        pred = self.ind.predict(x, verbose=0)[:, 0]
        pos_xy = x[pred > 0]
        neg_xy = x[pred < 0]
        sns.set()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            pos_xy[:, 1],
            pos_xy[:, 2],
            color=sns.color_palette()[0],
            label="Pos",
            alpha=0.5,
        )
        ax.scatter(
            neg_xy[:, 1],
            neg_xy[:, 2],
            color=sns.color_palette()[1],
            label="Neg",
            alpha=0.5,
        )
        ax.scatter(
            [s[1] for s in self.ce_buffer.s],
            [s[2] for s in self.ce_buffer.s],
            color="red",
            alpha=0.7,
            marker="x",
        )
        ax.scatter(
            [s[1] for s in self.ce_buffer.n],
            [s[2] for s in self.ce_buffer.n],
            color="red",
            alpha=0.7,
            marker="x",
        )
        if train:
            ax.scatter(
                self.train_x[self.train_y == 0, 1],
                self.train_x[self.train_y == 0, 2],
                color="tab:purple",
                alpha=0.7,
                marker="o",
                label="train unsafe",
            )
            ax.scatter(
                self.train_x[self.train_y == 1, 1],
                self.train_x[self.train_y == 1, 2],
                color="tab:cyan",
                alpha=0.7,
                marker="o",
                label="train safe",
            )
        for t in self._plot_trajectories:
            ax.plot(t[:, 1], t[:, 2], color="green", lw=0.8, alpha=0.9)

        if ce is not None:
            s_pre, s_post = ce
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

        ax.legend(loc="upper right")
        ax.set_title("Indicator image")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim([-self.env.bound_x, self.env.bound_x])
        ax.set_ylim([-1, self.env.bound_y])
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)


def policy_step(policy, x, deterministic=False):
    t = policy.predict(np.expand_dims(x, 0))[0]
    return np.argmax(t)


def single_trajectory(policy, env):
    done = False
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


def collect_trajectories(policy, env, n=100):
    buffer = []
    for i in range(n):
        buffer.extend(single_trajectory(policy, env))
    return np.stack(buffer, 0)


def sample_unsafe_region_car(env, n=500):
    unsafes = []
    while len(unsafes) < n:
        px = default_rng.integers(-env.bound_x, env.bound_x + 1)
        ax = default_rng.integers(-env.bound_x, env.bound_x + 1)
        ay = 0
        unsafes.append(np.array([px, ax, ay]))
    return np.stack(unsafes, axis=0)


def is_stable(policy, state):
    done = False
    x = env.reset(state)
    while not done:
        action = policy_step(policy, x)
        x, r, done, info = env.step(action)
        if r < -80:
            return False
    return True


def sample_bootstrap(policy, env, n=5000):
    print("Collecting unsafe data")
    buffer_pos = []
    buffer_neg = []
    while len(buffer_pos) < n or len(buffer_neg) < n:
        px = default_rng.integers(-env.bound_x, env.bound_x + 1)
        ax = default_rng.integers(-env.bound_x, env.bound_x + 1)
        ay = default_rng.integers(0, env.bound_y + 1)
        crash = np.abs(px - ax) <= env.safety_ball and ay <= 0
        state = np.array([px, ax, ay])
        obs = env.reset(state=state)
        if crash or (not is_stable(policy, state)):
            if len(buffer_neg) < n:
                print(f"neg: {len(buffer_neg)}/{n}")
                buffer_neg.append(obs)
        else:
            if len(buffer_pos) < n:
                print(f"pos: {len(buffer_pos)}/{n}")
                buffer_pos.append(obs)

    buffer_pos = np.stack(buffer_pos, axis=0)
    buffer_neg = np.stack(buffer_neg, axis=0)
    pos_labels = np.ones(buffer_pos.shape[0], dtype=np.int32)
    neg_labels = np.zeros(buffer_neg.shape[0], dtype=np.int32)

    x = np.concatenate([buffer_pos, buffer_neg], 0)
    y = np.concatenate([pos_labels, neg_labels], 0)
    return x, y


def get_indicator_training_data(policy, env):
    if args.bootstrap:
        return sample_bootstrap(policy, env)
    pos_samples = collect_trajectories(policy, env, n=args.p_trajectories)
    amount_of_unsafe = int(args.pm_ratio * pos_samples.shape[0])  # Same amount of less?
    neg_samples = sample_unsafe_region_car(env, amount_of_unsafe)

    pos_labels = np.ones(pos_samples.shape[0], dtype=np.int32)
    neg_labels = np.zeros(neg_samples.shape[0], dtype=np.int32)

    x = np.concatenate([pos_samples, neg_samples], 0)
    y = np.concatenate([pos_labels, neg_labels], 0)
    return x, y


# run_study()
parser = argparse.ArgumentParser()
parser.add_argument("--hidden", default=32, type=int)
parser.add_argument("--first_layer", action="store_true")
parser.add_argument("--bootstrap", action="store_true")
parser.add_argument("--single_step", action="store_true")
parser.add_argument("--pm_ratio", type=float, default=0.8)
parser.add_argument("--train_eps", type=float, default=0.02)
parser.add_argument("--p_trajectories", type=int, default=200)
parser.add_argument("--init_samples", type=int, default=0)

args = parser.parse_args()

env = CarEnv()

if args.first_layer:
    policy = ReluNetwork([6, 16, 3])
    policy(np.zeros((1, 3)))  # force weight creation
    policy.load_weights(f"weights/car_all.h5")
else:
    policy = ReluNetwork([16, 3])
    policy(np.zeros((1, 3)))  # force weight creation
    policy.load_weights(f"weights/car.h5")

loop = VerificationLoop(
    env,
    policy,
    ind_hidden_dim=args.hidden,
    init_samples=args.init_samples,
    train_eps=args.train_eps,
    gen_plot=False,
    first_layer_trick=args.first_layer,
)
eps_solved = -1
runtimes = []
epss = []
set_accuracy = []
for i, bayes_eps in enumerate(
    [
        0,
        1e-10,
        1e-3,
        1e-2,
        2e-2,
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
        timeout=2 * 60 * 60, max_steps=1 if args.single_step else None
    )
    if success:
        runtimes.append(time.time() - start_time)
        epss.append(bayes_eps)
        eps_solved = bayes_eps
        set_accuracy.append(loop.eval_init_accuracy())

    text = "SUCCESS!" if success else "FAIL"
    print(f"Verification loop [{i}] of car with eps={bayes_eps:0.2g} results={text}")

    if not success:
        break
print(f"eps_solved={eps_solved}")
result_dict = {
    "bayes_eps": epss,
    "runtimes": runtimes,
    "set_accuracy": set_accuracy,
    "args": vars(args),
    "last_time": time.time() - start_time,
}

os.makedirs("results", exist_ok=True)
for i in range(1000):
    filename = f"results/car_{i:03d}.json"
    if not os.path.isfile(filename):
        with open(filename, "a") as f:
            json.dump(result_dict, f)
        break