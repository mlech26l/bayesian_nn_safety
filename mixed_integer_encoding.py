import numpy as np
import os
import tensorflow as tf
import mip


class MinimizeLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred, sample_weight=None):
        print(
            "minimize loss called with y_true={}, y_pred={}".format(
                str(y_true), str(y_pred)
            )
        )
        return tf.reduce_mean(y_pred)


def add_l1_constraint(opt_model, var1, var2, dist):
    opt_model += var1 - var2 <= dist
    opt_model += var1 - var2 >= -dist


class ReluNetwork(tf.keras.Model):
    def __init__(self, layers):
        self._layer_spec = layers
        super(ReluNetwork, self).__init__()

        self.dense_layers = []
        for i, l in enumerate(self._layer_spec):
            activation = None if i == len(self._layer_spec) - 1 else "relu"
            self.dense_layers.append(tf.keras.layers.Dense(l, activation=activation))

    def call(self, inputs, **kwargs):
        x = inputs
        for c in self.dense_layers:
            x = c(x)
        return x


class MixedIntegerEncoding:
    def __init__(self, relu_network, input_lb, input_ub):
        self.relu_network = relu_network
        self.input_lb = np.array(input_lb)
        self.input_ub = np.array(input_ub)

    def encode_network(self, opt_model=None, name="", input_vars=None, bayes_eps=0.0):
        self.opt_model = mip.Model() if opt_model is None else opt_model
        self._relu_binary_nodes = []
        self._name = name
        if input_vars is None:
            input_vars = self._create_input_vars()
        self.input_vars = input_vars

        self._debug_vars = []
        self._preact_debug = []
        x = self.input_vars

        for i, l in enumerate(self.relu_network.dense_layers):
            self._relu_binary_nodes.append([])
            w, b = l.get_weights()
            x = self._encode_dense_layer(
                x,
                w,
                b,
                relu=i < len(self.relu_network.dense_layers) - 1,
                layer_name="dense_{}".format(i),
                bayes_eps=bayes_eps if i > 0 else 0,  # Only second layer onward
            )
            self._debug_vars.append(x)

        self.output_vars = x
        return x

    def _create_input_vars(self):
        input_vars = [
            self.opt_model.add_var(
                var_type=mip.CONTINUOUS,
                lb=self.input_lb[i],
                ub=self.input_ub[i],
                name="{}input_{}".format(self._name, i),
            )
            for i in range(self.input_lb.shape[0])
        ]
        return input_vars

    def _encode_dense_layer(self, in_vars, w, b, relu, layer_name, bayes_eps):
        in_size = w.shape[0]
        out_size = w.shape[1]
        # print("[DEBUG]: w.shape: {}, in_vars: {}".format(str(w.shape), len(in_vars)))
        assert len(in_vars) == in_size

        out_vars = []
        for o in range(out_size):
            lb = b[o]
            ub = b[o]
            accumulator_list = []
            second_accumulator_list = []
            for i in range(in_size):
                in_w = w[i, o]
                in_v = in_vars[i]

                if in_v is None:
                    continue

                in_bounds = [in_w * in_v.lb, in_w * in_v.ub]
                if bayes_eps > 0:
                    assert in_v.lb >= 0, "Hol'up! Something ain't right"
                    in_bounds = [
                        (in_w - bayes_eps) * in_v.ub,
                        (in_w - bayes_eps) * in_v.lb,
                        (in_w + bayes_eps) * in_v.ub,
                        (in_w + bayes_eps) * in_v.lb,
                    ]
                lb += np.min(in_bounds)
                ub += np.max(in_bounds)
                if np.abs(in_w) > 1e-7:
                    if bayes_eps > 0:
                        accumulator_list.append(in_v * (in_w - bayes_eps))
                        second_accumulator_list.append(in_v * (in_w + bayes_eps))
                    else:
                        accumulator_list.append(in_v * in_w)

            if relu and ub <= 0:
                # Relu inactive
                out_vars.append(None)
                continue
            # Now we have the bounds of the neuron
            pre_act_neuron = self.opt_model.add_var(
                var_type=mip.CONTINUOUS,
                lb=lb,
                ub=ub,
                name="{}{}_{}".format(self._name, layer_name, o),
            )

            # Create variable and add equality constrain
            if bayes_eps > 0:
                # In Bayes setting we have only lower and upper bounds on the pre-act neuron

                accumulator_list.append(pre_act_neuron * -1)
                second_accumulator_list.append(pre_act_neuron * -1)
                self.opt_model += mip.xsum(accumulator_list) <= -b[o]
                self.opt_model += mip.xsum(second_accumulator_list) >= -b[o]
            else:
                # In deterministic setting we have equality
                accumulator_list.append(pre_act_neuron * -1)
                self.opt_model += mip.xsum(accumulator_list) == -b[o]
            self._preact_debug.append(pre_act_neuron)
            if relu and lb < 0:
                # Encode non-linearity
                post_act_neuron = self._encode_relu_node(pre_act_neuron, lb, ub)
            else:
                # Just a linear layer (=last layer)
                post_act_neuron = pre_act_neuron
            out_vars.append(post_act_neuron)
        return out_vars

    def _encode_relu_node(self, pre_act_neuron, lb, ub):
        binary_var = self.opt_model.add_var(
            var_type=mip.BINARY,
            # lowlbBound=0,
            # upBound=1,
            name="{}_binary".format(pre_act_neuron.name),
        )
        self._relu_binary_nodes[-1].append(binary_var)
        # y >= 0
        post_act_neuron = self.opt_model.add_var(
            var_type=mip.CONTINUOUS,
            lb=0,
            ub=np.max([ub, 0]),
            name="{}_relu".format(pre_act_neuron.name),
        )

        # y >= x
        self.opt_model += post_act_neuron - pre_act_neuron >= 0

        # y <= ub*d
        self.opt_model += ub * binary_var - post_act_neuron >= 0

        # y <= x - lb*(1-d)
        self.opt_model += pre_act_neuron + binary_var * lb + post_act_neuron * -1 >= lb
        return post_act_neuron

    def constrain_output_class(self, class_id):
        for i, v in enumerate(self.output_vars):
            if i == class_id:
                continue
            self.opt_model += self.output_vars[i] - v >= 0
            # )

    def bind_binary_vars_to_relu(self, x):
        mapping = []
        x = np.expand_dims(x, axis=0)
        outputs, hidden_layers = self.relu_network.call_and_collect(x)
        # print("len(_relu_binary_nodes):", str(len(self._relu_binary_nodes)))
        # for i in range(len(self._relu_binary_nodes)):
        #     print("len[{}]: {}".format(i, len(self._relu_binary_nodes[i])))
        for i in range(len(hidden_layers) - 1):
            hidden_values = hidden_layers[i].numpy()[0].flatten()
            # print("len(_relu_binary_nodes[i]):", str(len(self._relu_binary_nodes[i])))
            # print("hidden_values.shape:", str(hidden_values.shape))

            for j in range(hidden_values.shape[0]):
                if self._relu_binary_nodes[i][j] is None:
                    continue
                bin_value = int(hidden_values[j] > 0)
                mapping.append((self._relu_binary_nodes[i][j], bin_value))

        outputs = outputs.numpy()[0]

        for (index_list, bin_vars) in self._relu_binary_nodes[-1]:
            argmax = np.argmax([outputs[i] for i in index_list])
            for i, index in enumerate(index_list):
                bin_value = int(i == argmax)
                mapping.append((bin_vars[i], bin_value))

        return mapping

    def add_output_max_pooling(self, index_list, name):
        var_list = [self.output_vars[i] for i in index_list]
        lbs = [v.lb for v in var_list]
        ubs = [v.ub for v in var_list]
        max_node = self.opt_model.add_var(lb=np.max(lbs), ub=np.max(ubs), name=name)

        binary_vars = []
        for i, v in enumerate(var_list):
            self.opt_model += max_node - v >= 0
            binary_vars.append(
                self.opt_model.add_var(
                    var_type=mip.BINARY, name="{}_{}_binary".format(name, i)
                )
            )
            u1k = np.max([ubs[j] for j in range(len(var_list)) if i != j])
            termumaxi = u1k - lbs[i]

            self.opt_model += (
                -max_node + var_list[i] - termumaxi * binary_vars[i] >= -termumaxi
            )
        self._relu_binary_nodes[-1].append((index_list, binary_vars))
        self.opt_model += mip.xsum(binary_vars) == 1
        return max_node, binary_vars

    def _get_var_values(self, var_list):
        var_values = []
        for x in var_list:
            var_values.append(float(x.x))
        return np.array(var_values).astype(np.float32)

    def get_input_assignment(self):
        return self._get_var_values(self.input_vars)

    def get_allowed_vs_others_objective(self, allowed_classes):
        forbidden_classes = [
            i for i in range(len(self.output_vars)) if i not in allowed_classes
        ]

        allowed_maxes = self.add_output_max_pooling(
            allowed_classes, name="{}max_allowed".format(self._name)
        )
        forbidden_maxes = self.add_output_max_pooling(
            forbidden_classes, name="{}max_forbidden".format(self._name)
        )
        # This should be minimized
        # if it's negative the spec are violated!
        return allowed_maxes - forbidden_maxes


def encode_alpha_clip(opt_model, x, alpha):
    lb = x.lb
    ub = x.ub
    y = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=alpha,
        ub=ub,
        name="{}_alpha".format(x.name),
    )
    a = opt_model.add_var(var_type=mip.BINARY, lb=0, ub=1, name="a_{}".format(x.name))

    opt_model += y - x >= 0
    opt_model += y - (ub - alpha) * a <= alpha
    opt_model += y - x - lb * a <= -lb
    return y


def encode_beta_clip(opt_model, x, beta):
    lb = x.lb
    ub = x.ub
    y = opt_model.add_var(
        var_type=mip.CONTINUOUS,
        lb=lb,
        ub=beta,
        name="{}_alpha".format(x.name),
    )
    b = opt_model.add_var(
        var_type=mip.BINARY,
        lb=0,
        ub=1,
        name="b_{}".format(x.name),
    )

    opt_model += y - x <= 0
    opt_model += y + (-beta + lb) * b >= lb
    opt_model += y - x + ub * b >= 0
    return y


def clip_at_pm1(opt_model, x):
    v1 = encode_alpha_clip(opt_model, x, -1)
    v2 = encode_beta_clip(opt_model, v1, 1)
    return v2


def clip_at_pm_k(opt_model, x, k):
    v1 = encode_alpha_clip(opt_model, x, -k)
    v2 = encode_beta_clip(opt_model, v1, k)
    return v2


if __name__ == "__main__":
    network = ReluNetwork([32, 16, 2])
    train_data_x = np.random.normal(size=[200, 4])
    train_data_y = np.random.normal(size=[200, 2])
    network.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    network.fit(x=train_data_x, y=train_data_y, batch_size=50, epochs=5)

    lb, ub = np.zeros(4), np.ones(4)

    input_sample = np.random.normal(size=[1, 4]).astype(np.float32)
    y = network(input_sample)[0].numpy()
    print("TF network output: ", y)

    epsilon = 1e-3
    mip_model = MixedIntegerEncoding(
        network, input_sample[0] - epsilon, input_sample[0] + epsilon
    )
    mip_model.encode_network()

    mip_model.opt_model.objective = mip.maximize(mip.xsum(mip_model.output_vars))

    status = mip_model.opt_model.optimize()
    print("opt_value {:0.2f}".format(mip_model.opt_model.objective_value))

    solution = []
    if status == mip.OptimizationStatus.OPTIMAL:
        print(
            "optimal solution cost {} found".format(mip_model.opt_model.objective_value)
        )
    elif status == mip.OptimizationStatus.FEASIBLE:
        print(
            "sol.cost {} found, best possible: {}".format(
                mip_model.opt_model.objective_value, mip_model.opt_model.objective_bound
            )
        )
    elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
        print(
            "no feasible solution found, lower bound is: {}".format(
                mip_model.opt_model.objective_bound
            )
        )
        print("THIS SHOULD NEVER HAPPEN")
    if (
        status == mip.OptimizationStatus.OPTIMAL
        or status == mip.OptimizationStatus.FEASIBLE
    ):
        for i, v in enumerate(mip_model.output_vars):
            print("var[{}]: {:0.3f}".format(i, v.x))