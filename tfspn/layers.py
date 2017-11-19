'''
Created on May 20, 2017

@author: stelzner
'''
import tensorflow as tf
import tensorflow.contrib.distributions as dist
import numpy as np
import tfspn.tfspn as tfspn
from collections import defaultdict

# Implementing a layered SPN
# Goal: reducing the number of tensors and variables in an SPN
# to accelerate weight learning and TF initialization.
# Therefore: group nodes to layers, and have one high-dimensional tensor per layer

# datatype used in SPN
spn_type = tf.float64


# default reduction of input layers - just concat features
def concat_layers(inputs):
    return tf.concat(inputs, 1)


class Layer:
    def __init__(self):
        self.scope = None
        self.last_output = None
        self.nodes = None

    def value(self, x, marginalized=None):
        """ Return the output tensor of the layer
        x - input tensor of shape [datapoints, features]
        marginalized - list of bools determining if a variable is marginalized """
        raise NotImplementedError("Abstract method")


class ProductLayer(Layer):
    def __init__(self, input_layers, structure, reduction=concat_layers):
        """ Create product layer
        input_layers - list of layers that serve as input
        structure - numpy array of shape (in_dim, out_dim) that determines the structure
                    structure[i, j] == 1 means input i is connected to output j
        reduction - function to turn input layer values into actual input tensor"""
        super().__init__()
        self.reduction = reduction
        self.inputs = input_layers
        self.structure_np = structure
        self.structure = tf.constant(structure, dtype=spn_type)
        self.in_dim, self.out_dim = structure.shape

    def value(self, x, marginalized=None):
        input_values = [inp.value(x, marginalized) for inp in self.inputs]
        input_tensor = self.reduction(input_values)
        self.last_output = tf.matmul(input_tensor, self.structure)
        return self.last_output

    def map_value(self, x, evidence, sample=False):
        child_probs = []
        child_values = []
        for child in self.inputs:
            v, p = child.map_value(x, evidence, sample)
            child_values.append(v)
            child_probs.append(p)
        # TODO decide if reduction funciton makes sense
        child_prob_tensor = self.reduction(child_probs)
        child_value_tensor = self.reduction(child_values)
        n = tf.shape(child_value_tensor)[0]
        f = tf.shape(child_value_tensor)[2]
        probs = tf.matmul(child_prob_tensor, self.structure)
        # structure_copies = tf.tile(self.structure, [tf.shape(child_value_tensor)[2], 1])
        # structure_copy_shape = tf.concat([tf.shape(self.structure), tf.shape(child_value_tensor)[2:3]], axis=0)
        # structure_copies = tf.reshape(structure_copies, structure_copy_shape)
        # structure_copies = tf.transpose(structure_copies, [2, 0, 1])
        # To get the output value, multiply each [:, :, i] of the value tensor with the structure.
        # This is achieved by conflating the instance and feature dimensions using reshape,
        # doing a single matmul, and then reshaping again.
        child_value_tensor = tf.transpose(child_value_tensor, [0, 2, 1])
        child_value_tensor = tf.reshape(child_value_tensor, [n * f, self.in_dim])
        values = tf.matmul(child_value_tensor, self.structure)
        values = tf.transpose(tf.reshape(values, [n, f, self.out_dim]), [0, 2, 1])
        return values, probs

    def network_size(self):
        layers, nodes, variables = 1, self.out_dim, self.in_dim * self.out_dim
        for child in self.inputs:
            if not isinstance(child, Layer):
                print(child)
            l, n, v = child.network_size()
            layers += l
            nodes += n
            variables += v
        return layers, nodes, variables

    def var_list(self):
        result = []
        for child in self.inputs:
            result.extend(child.var_list())
        return result

    def validate(self, exception=False):
        if self.scope is not None:
            return True
        input_scopes = []
        for inp in self.inputs:
            if not inp.validate(exception):
                if exception:
                    raise RuntimeError()
                return False
            input_scopes += inp.scope

        self.scope = []
        for col in range(self.out_dim):
            self.scope.append(set([]))
            for row in range(self.in_dim):
                if self.structure_np[row, col] == 1:
                    union = self.scope[-1] | input_scopes[row]
                    if len(union) != len(self.scope[-1]) + len(input_scopes[row]):
                        if exception:
                            raise RuntimeError()
                        return False
                    self.scope[-1] = union
        return True

    def to_nodes(self):
        self.nodes = []
        all_children = []
        for child in self.inputs:
            if child.nodes is None:
                child.to_nodes()
            all_children.extend(child.nodes)

        for i in range(self.out_dim):
            children = []
            for j in range(self.in_dim):
                if self.structure_np[j, i] > 0:
                    children.append(all_children[j])
            self.nodes.append(tfspn.ProductNode(str(id(self)) + str(i), *children))


class SumLayer(Layer):
    SAMPLING_RELAXATION = 0.5

    def __init__(self, input_layers, structure, reduction=concat_layers):
        """ Create sum layer
        input_layers - list of layers that serve as input
        structure - numpy array of dimensions [in_dim, out_dim] that determines the structure
                    structure[i, j] == w > 0 means input i is connected to output j with weight w
        reduction - function to turn input layer values into actual input tensor"""
        super().__init__()
        self.reduction = reduction
        self.inputs = input_layers

        # self.weights is the variable containing log of actual weights
        self.weights_np = np.array(structure)
        for x in np.nditer(self.weights_np, op_flags=['readwrite']):
            x[...] = np.log(x) if x > 0 else np.NINF
        self.weights = tf.Variable(self.weights_np, trainable=True, dtype=spn_type)

        # self.structure is binary mask. non-existent connections set to negative infinity.
        self.structure_np = np.array(structure)
        self.structure = np.array(structure)
        for x in np.nditer(self.structure, op_flags=['readwrite']):
            x[...] = 0 if x > 0 else np.NINF
        self.structure = tf.constant(self.structure, dtype=spn_type)

        self.in_dim, self.out_dim = structure.shape

    def value(self, x, marginalized=None):
        input_values = [inp.value(x, marginalized) for inp in self.inputs]
        input_tensor = tf.expand_dims(self.reduction(input_values), 2)
        # Using broadcasting to add input_tensors out_dim-times
        # self.structure and self.weights have shape [in_dim, out_dim]
        # input_tensor has shape [datapoints, in_dim, 1]
        weights_added = (self.structure + self.weights) + input_tensor
        # weights add has shape [datapoints, in_dim, out_dim]
        self.last_output = tf.reduce_logsumexp(weights_added, 1, keep_dims=False)
        # output has shape [datapoints, out_dim]
        return self.last_output

    def map_value(self, x, evidence, sample=False):
        child_values = []
        child_probs = []
        for child in self.inputs:
            v, p = child.map_value(x, evidence, sample)
            child_values.append(v)
            child_probs.append(p)
        child_prob_tensor = tf.expand_dims(self.reduction(child_probs), 2)
        child_values_tensor = self.reduction(child_values)
        weights_added = (self.structure + self.weights) + child_prob_tensor
        # probs = tf.reduce_logsumexp(weights_added, 1, keep_dims=False)
        if not sample:
            probs = tf.reduce_max(weights_added, 1, keep_dims=False)
            max_idx = tf.cast(tf.argmax(weights_added, axis=1), tf.int32)
            n = tf.shape(child_values_tensor)[0]
            idx_x = tf.tile(tf.range(0, n, 1), [self.out_dim])
            idx_x = tf.transpose(tf.reshape(idx_x, [self.out_dim, n]))
            values = tf.gather_nd(child_values_tensor, tf.stack([idx_x, max_idx], axis=2))
        else:
            weights_added_tr = tf.cast(tf.transpose(weights_added, [0, 2, 1]), tf.float32)
            # cat_dists = dist.Categorical(logits=weights_added)
            relaxed_dists = dist.RelaxedOneHotCategorical(SumLayer.SAMPLING_RELAXATION,
                                                          logits=weights_added_tr)
            idxs = relaxed_dists.sample()
            idxs = tf.cast(idxs, spn_type)
            # print(idxs, child_values_tensor)
            values = tf.matmul(idxs, child_values_tensor)
            # FIXME find better way to prevent nans from log(0)
            probs = tf.log(idxs + 1e-20) + tf.cast(weights_added_tr, spn_type)
            probs = tf.reduce_logsumexp(probs, axis=2)
            # max_idx = cat_dists.sample()
            # probs = cat_dists.log_prob(max_idx)

        return values, probs

    def network_size(self):
        layers, nodes, variables = 1, self.out_dim, self.in_dim * self.out_dim
        for child in self.inputs:
            l, n, v = child.network_size()
            layers += l
            nodes += n
            variables += v
        return layers, nodes, variables

    def var_list(self):
        result = [self.weights]
        for child in self.inputs:
            result.extend(child.var_list())
        return result

    def validate(self, exception=False):
        if self.scope is not None:
            return True
        input_scopes = []
        for inp in self.inputs:
            if not inp.validate(exception):
                if exception:
                    raise RuntimeError()
                return False
            input_scopes += inp.scope

        self.scope = []
        for col in range(self.out_dim):
            self.scope.append(set([]))
            for row in range(self.in_dim):
                if self.structure_np[row, col] > 0:
                    if len(self.scope[-1]) == 0:
                        self.scope[-1] = input_scopes[row]
                    elif self.scope[-1] != input_scopes[row]:
                        if exception:
                            raise RuntimeError
                        return False
        return True

    def to_nodes(self):
        self.nodes = []
        all_children = []
        for child in self.inputs:
            if child.nodes is None:
                child.to_nodes()
            all_children.extend(child.nodes)

        for i in range(self.out_dim):
            children = []
            for j in range(self.in_dim):
                if self.structure_np[j, i] > 0:
                    children.append(all_children[j])
            self.nodes.append(tfspn.SumNode(str(id(self)) + str(i),
                                            [1 / len(children)] * len(children),
                                            *children))


class LeafLayer(Layer):
    def __init__(self, feature_ids):
        super().__init__()
        self.feature_ids = feature_ids
        # @CLEANUP
        # Get rid of self.size
        self.size = len(feature_ids)
        self.out_dim = self.size

    def validate(self, exception=False):
        if self.scope is None:
            self.scope = [{feature} for feature in self.feature_ids]
        return True

    def value(self, x, marginalized=None):
        raise NotImplementedError('Abstract method')

    def map_points(self, sample=False):
        raise NotImplementedError('Abstract method')

    # Return MAP-configurations and their likelihoods
    # The configuration is a (n, d, f)-shaped vector, where n is the number of dataitems,
    # d the output dimension of the layer and f the dimension of the dataitems.
    def map_value(self, x, evidence, sample=False):
        n = tf.shape(x)[0]
        x_t = tf.transpose(x)
        # (n, self.out_dim)
        evidence_values = tf.transpose(tf.gather(x_t, self.feature_ids))
        map_values, map_probs = self.map_points(sample)
        map_values = tf.tile(map_values, [tf.shape(evidence_values)[0]])
        map_values = tf.reshape(map_values, tf.shape(evidence_values))
        evidence_gathered = tf.gather(evidence, self.feature_ids)
        evidence_repeated = tf.tile(evidence_gathered, [n])
        evidence_repeated = tf.reshape(evidence_repeated, tf.shape(map_values))
        evidence_probs = self.value(x)
        res_values = tf.where(evidence_repeated, evidence_values, map_values)
        map_probs = tf.reshape(tf.tile(map_probs, [n]), tf.shape(evidence_repeated))
        probs = tf.where(evidence_repeated, evidence_probs, map_probs)
        scatter_indices_x = tf.range(0, tf.shape(res_values)[1], 1)
        scatter_indices_x = tf.reshape(tf.tile(scatter_indices_x, [tf.shape(res_values)[0]]), tf.shape(res_values))
        scatter_indices_y = tf.range(0, tf.shape(res_values)[0], 1)
        scatter_indices_y = tf.tile(scatter_indices_y, [tf.shape(res_values)[1]])
        scatter_indices_y = tf.transpose(tf.reshape(scatter_indices_y, tf.reverse(tf.shape(res_values), axis=[0])))
        scatter_indices_z = tf.reshape(tf.tile(tf.constant(self.feature_ids), [n]), tf.shape(res_values))
        scatter_indices = tf.stack([scatter_indices_y, scatter_indices_x, scatter_indices_z], axis=2)
        result_shape = tf.concat([tf.shape(res_values), tf.shape(x)[1:2]], axis=0)
        result = tf.scatter_nd(scatter_indices, res_values, shape=result_shape)
        return result, probs

    def network_size(self):
        return 1, self.size, self.size


class GaussianLayer(LeafLayer):
    # Lower bound for standard deviations. Needs to be > 0, otherwise
    # the density function is unbounded, and so is the generative loss
    MIN_STDEV = 0.01

    def __init__(self, feature_ids, means=None, stdevs=None):
        super().__init__(feature_ids)
        if means is None:
            self.means = tf.random_normal([self.size], dtype=spn_type)
        else:
            self.means = tf.constant(means, dtype=spn_type)
        self.means = tf.Variable(self.means, trainable=True, dtype=spn_type)

        if stdevs is None:
            self.stdevs = tf.random_gamma([self.size], alpha=2, beta=2, dtype=spn_type)
        else:
            self.stdevs = tf.constant(stdevs, dtype=spn_type)
        self.stdevs = tf.Variable(tf.log(self.stdevs), trainable=True, dtype=spn_type)

        self.distributions = dist.Normal(self.means, tf.exp(self.stdevs) + GaussianLayer.MIN_STDEV)

    def value(self, x, marginalized=None):
        # TODO find better way to gather columns than to do in-memory transpose
        x = tf.transpose(x)
        inp = tf.gather(x, self.feature_ids)
        inp = tf.transpose(inp)
        self.last_output = self.distributions.log_prob(inp)
        if marginalized is not None:
            marginalized_idx = 1 - tf.expand_dims(tf.gather(tf.cast(marginalized, tf.float64), self.feature_ids), 0)
            self.last_output = self.last_output * marginalized_idx
        return self.last_output

    def map_points(self, sample=False):
        if sample:
            values = self.distributions.sample()
            probs = self.distributions.log_prob(values)
            return values, probs
        return self.means, self.distributions.log_prob(self.means)

    def to_nodes(self):
        self.nodes = []
        # means, stdevs = tf.Session().run([self.means, self.stdevs])
        for i in range(self.size):
            self.nodes.append(tfspn.GaussianNode(str(id(self)) + str(i),
                                                 self.feature_ids[i],
                                                 str(i),
                                                 np.random.normal(),
                                                 np.random.gamma(1, 1)))

    def var_list(self):
        return [self.means, self.stdevs]


class BernoulliLayer(LeafLayer):
    def __init__(self, feature_ids, probs=None):
        super().__init__(feature_ids)
        if probs is None:
            self.probs = tf.random_uniform([len(feature_ids)], minval=0, maxval=1, dtype=spn_type)
        else:
            self.probs = tf.constant(probs, dtype=spn_type)
        self.probs = tf.Variable(tf.log(self.probs), trainable=True, dtype=spn_type)
        self.distributions = dist.Bernoulli(logits=self.probs)

    def value(self, x, marginalized=None):
        # TODO find better way to gather columns than to do in-memory transpose
        x = tf.transpose(x)
        inp = tf.gather(x, self.feature_ids)
        inp = tf.transpose(inp)
        self.last_output = self.distributions.log_prob(inp)
        if marginalized is not None:
            marginalized_idx = 1 - tf.expand_dims(tf.gather(tf.cast(marginalized, tf.float64), self.feature_ids), 0)
            self.last_output = self.last_output * marginalized_idx
        return self.last_output

    def map_points(self, sample=False):
        if sample:
            values = tf.cast(self.distributions.sample(), spn_type)
            probs = self.distributions.log_prob(values)
            return values, probs
        real_probs = tf.exp(self.probs)
        zeros = tf.cast(tf.fill([self.size], 0), spn_type)
        ninfs = tf.cast(tf.fill([self.size], np.NINF), spn_type)
        values = tf.where(real_probs > 0.5, zeros, ninfs)
        probs = tf.maximum(real_probs, 1 - real_probs)
        return values, tf.log(probs)

    def to_nodes(self):
        self.nodes = []
        # ps = tf.Session().run(tf.exp(self.probs))
        for i in range(self.size):
            self.nodes.append(tfspn.BernoulliNode(str(id(self)) + str(i),
                                                  self.feature_ids[i],
                                                  str(i),
                                                  0.5))

    def var_list(self):
        return [self.probs]


# Leaf layer whose probability value is one if the features match
# the given values and (almost) zero otherwise
class DiscreteLayer(LeafLayer):
    def __init__(self, feature_ids, values):
        super().__init__(feature_ids)
        self.values_np = np.array(values)
        self.values = tf.constant(values, dtype=spn_type)

    def value(self, x, marginalized=None):
        x = tf.transpose(x)
        inp = tf.gather(x, self.feature_ids)
        inp = tf.transpose(inp)
        # Value can not be set to negative infinity (probability zero), because we
        # multiply by zero (here and in product nodes).
        # However, log prob -1e20 is plenty small.
        ninf_tensor = tf.cast(tf.fill(dims=tf.shape(inp), value=-1e20), spn_type)
        self.last_output = tf.where(tf.equal(self.values, inp), tf.zeros_like(inp), ninf_tensor)
        if marginalized is not None:
            marginalized_idx = 1 - tf.expand_dims(tf.gather(tf.cast(marginalized, tf.float64), self.feature_ids), 0)
            self.last_output = self.last_output * marginalized_idx
        return self.last_output

    def map_points(self, sample=False):
        # There is only one possible value, so no sampling required
        return self.values, tf.cast(tf.fill([self.size], 0), spn_type)

    def to_nodes(self):
        self.nodes = []
        for i in range(self.size):
            # TODO create better match for discrete layer
            self.nodes.append(tfspn.DiscreteNode(str(id(self)) + str(i),
                                                  self.feature_ids[i],
                                                  str(i),
                                                  self.values_np[i]))

    def var_list(self):
        return []


# discriminative_cost returns a tensor that can be minimized for discriminative learning
# this is done by maximizing P(label | features)
# - inp: placeholder for the input data
# - features: list of one bool per feature, True if that feature is a label
# - layer: top layer of the SPN that should be trained. Output dim should be 1.
def discriminative_cost(inp, labels, layer):
    joined_prob = layer.value(inp)
    marginal_prob = layer.value(inp, labels)
    conditional_prob = joined_prob - marginal_prob
    cost = -1 * tf.reduce_sum(conditional_prob, 0)
    return cost


def translate_nodes_to_layers(root, max_layer_size=100):
    return NodesToLayers(root, max_layer_size).root_layer


class LayerShell:
    def __init__(self, nodes, id):
        self.nodes = nodes
        self.id = id


class NodesToLayers:
    def __init__(self, root, max_layer_size=100):
        self.all_nodes = []
        self.leafs = []
        self.indeg = {}
        self.visited = set()
        self.parents = {}
        self.levels = []
        self.root = root
        self.nodes_to_layers = {}
        self.start_visit(root, self.init_node)
        self.build_levels()
        self.layers = []
        self.final_layers = []
        self.cur_layer_id = 0

        for level in self.levels:
            print(len(level))

        self.register_layer([root])
        for i in range(1, len(self.levels)):
            self.process_level(i, max_layer_size)

        self.finalize_layers()
        self.root_layer = self.final_layers[-1]

    def finalize_layers(self):
        layer_id_to_layer = {}
        for layer_shell in reversed(self.layers):
            ex_node = layer_shell.nodes[0]
            if isinstance(ex_node, tfspn.GaussianNode):
                layer = GaussianLayer([n.featureIdx for n in layer_shell.nodes],
                                      [n.mean for n in layer_shell.nodes],
                                      [n.stdev for n in layer_shell.nodes])
            elif isinstance(ex_node, tfspn.BernoulliNode):
                layer = BernoulliLayer([n.featureIdx for n in layer_shell.nodes],
                                       [n.p for n in layer_shell.nodes])
            elif isinstance(ex_node, tfspn.DiscreteNode):
                layer = DiscreteLayer([n.featureIdx for n in layer_shell.nodes],
                                      [n.value for n in layer_shell.nodes])
            elif isinstance(ex_node, tfspn.SumNode) or isinstance(ex_node, tfspn.ProductNode):
                input_layers = set()
                for node in layer_shell.nodes:
                    for child in node.children:
                        child_layer_id = self.nodes_to_layers[child].id
                        child_layer = layer_id_to_layer[child_layer_id]
                        input_layers.add(child_layer)
                input_layers = list(input_layers)
                input_index = [0]
                for i in range(1, len(input_layers)):
                    input_index.append(input_index[-1] + input_layers[i - 1].out_dim)
                total_in_dim = input_index[-1] + input_layers[-1].out_dim
                structure = np.zeros((total_in_dim, len(layer_shell.nodes)))
                for i, node in enumerate(layer_shell.nodes):
                    for j, child in enumerate(node.children):
                        child_layer_id = self.nodes_to_layers[child].id
                        child_layer = layer_id_to_layer[child_layer_id]
                        child_layer_num = -1
                        # @EFFICIENCY create map {layer: input_idx} instead of searching
                        for k in range(len(input_layers)):
                            if input_layers[k] == child_layer:
                                child_layer_num = k
                                break
                        # @EFFICIENCY create map {node in layerShell: index} instead of searching
                        child_offset = -1
                        for k in range(len(self.nodes_to_layers[child].nodes)):
                            if child == self.nodes_to_layers[child].nodes[k]:
                                child_offset = k
                                break
                        absolute_child_index = input_index[child_layer_num] + child_offset
                        if isinstance(ex_node, tfspn.SumNode):
                            structure[absolute_child_index, i] = node.weights[j]
                        else:
                            structure[absolute_child_index, i] = 1
                if isinstance(ex_node, tfspn.SumNode):
                    layer = SumLayer(input_layers, structure)
                else:
                    layer = ProductLayer(input_layers, structure)
            else:
                raise NotImplementedError()
            layer_id_to_layer[layer_shell.id] = layer
            self.final_layers.append(layer)

    def start_visit(self, node, callback):
        self.visited = set()
        self.visit(node, callback)

    def visit(self, node, callback):
        callback(node)
        self.visited.add(node)
        if not node.leaf:
            for child in node.children:
                if child not in self.visited:
                    self.visit(child, callback)

    def init_node(self, node):
        self.all_nodes.append(node)
        if node not in self.parents:
            self.parents[node] = []
        if not node.leaf:
            self.indeg[node] = len(node.children)
            for child in node.children:
                if child not in self.parents:
                    self.parents[child] = []
                self.parents[child].append(node)
        else:
            self.leafs.append(node)
            self.indeg[node] = 0

    def build_levels(self):
        indeg = self.indeg.copy()
        current_level = self.leafs.copy()
        while len(current_level) > 0:
            self.levels.append(current_level)
            next_level = []
            for node in current_level:
                for parent in self.parents[node]:
                    indeg[parent] -= 1
                    if indeg[parent] == 0:
                        next_level.append(parent)
            current_level = next_level
        self.levels = list(reversed(self.levels))

    def register_layer(self, nodes):
        layer = LayerShell(nodes, self.cur_layer_id)
        self.cur_layer_id += 1
        for node in nodes:
            self.nodes_to_layers[node] = layer
        self.layers.append(layer)

    def process_level(self, idx, max_layer_size):
        nodes_to_parent_layers = {}
        partitions = defaultdict(list)
        for node in self.levels[idx]:
            nodes_to_parent_layers[node] = frozenset([self.nodes_to_layers[parent].id
                                                     for parent in self.parents[node]])
            partitions[type(node)].append(node)
        partitions = list(partitions.values())
        for part in partitions:
            if len(part) > max_layer_size:
                groups = defaultdict(list)
                for node in part:
                    groups[nodes_to_parent_layers[node]].append(node)
                groups = list(groups.values())
                for group in groups:
                    cur_idx = max_layer_size
                    while cur_idx <= len(group):
                        self.register_layer(group[cur_idx - max_layer_size:cur_idx])
                        cur_idx += max_layer_size
                    if cur_idx - max_layer_size != len(group):
                        self.register_layer(group[cur_idx - max_layer_size:])
            else:
                self.register_layer(part)





