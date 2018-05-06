'''
Created on Dec 15, 2016

@author: alejandro molina
@author: antonio vergari
'''

from _collections import deque
import logging
import sys
from time import perf_counter

from numba import jit
from numba.tests.npyufunc.test_ufunc import dtype
from numpy import float64
from scipy.special import logit
import scipy.stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

from mlutils.statistics import gaussianpdf, nplogpoissonpmf, \
    nplogbernoullipmf_fast
from mlutils.statistics import logbernoullipmf_fast, bernoullipmf
from mlutils.statistics import logpoissonpmf, poissonpmf
from mlutils.statistics import gammapdf, betapdf
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as distributions
from tfspn.piecewise import is_piecewice_linear_pdf, two_staged_sampling_piecewise_linear


# log of zero const, to avoid -inf
# numpy.exp(LOG_ZERO) = 0
LOG_ZERO = -1e3
RAND_SEED = 1337


def IS_LOG_ZERO(log_val):
    """
    checks for a value to represent the logarithm of 0.
    The identity to be verified is that:
    IS_LOG_ZERO(x) && exp(x) == 0
    according to the constant LOG_ZERO
    """
    return (log_val <= LOG_ZERO)


# defining a numerical correction for 0
EPSILON = sys.float_info.min


def GetCreateVar(initVal, scope, context, name, dtype="float64"):
    # search for it, if it doesn't exist create it

    vs = tf.get_default_graph().get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + context + "/" + name)
    # print("Q", scope+context+"/"+name, len(vs))
    # vs = []
    # for v in vs:
    #    print(v.name)
    # print("ddddddddddddddddddddddd")

    if len(vs) == 0:
        var = tf.Variable(initVal, dtype=dtype, name=name)
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # vs = tf.get_default_graph().get_collection(tf.GraphKeys.VARIABLES, scope="")
        # for v in vs:
        #    print(v.name)
        # print("bbbbbbbbbbbbbbbbbbbbbbbbbbb")
        # print(scope, context,name, "created")
        return var
    elif len(vs) == 1:
        # print(scope, context,name, "reused")
        return vs[0]
    else:
        assert False, "More than one variable found %s %s" % (scope, name)


def GetVar(scope, context, name):
    # search for it, if it doesn't exist create it

    vs = tf.get_default_graph().get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + context + "/" + name + ":0")

    # print("Q", scope+context+"/"+name, len(vs))
    # for v in vs:
    #    print(v.name)
    if len(vs) == 0:
        assert False, "No variable found %s %s" % (scope, name)
    elif len(vs) == 1:
        return vs[0]
    else:
        return vs


class Node(object):

    id_counter = 0

    def __init__(self):
        self.id = Node.id_counter
        Node.id_counter += 1

        self.label = ''

    def top_down_nodes(self):
        nodes_to_process = deque()
        nodes_to_process.append(self)
        visited_nodes = set()

        while nodes_to_process:
            n = nodes_to_process.popleft()
            if n not in visited_nodes:
                yield n
                nodes_to_process.extend(n.children)
                visited_nodes.add(n)

    def get_leaves(self):
        return [n for n in self.top_down_nodes() if len(n.children) == 0]

    def __hash__(self):
        """
        A node has a unique id
        """
        # return hash(self.label)
        return hash(self.id)

    def __eq__(self, other):
        """
        WRITEME
        """
        return self.id == other.id

    def __set_serializable__(self):
        self.Serializable_attrs = [k for k in self.__dict__.keys()]

    def __getstate__(self):
        return {a: getattr(self, a) for a in self.Serializable_attrs}


class ProductNode(Node):

    def __init__(self, name, *nodes):
        Node.__init__(self)
        self.label = "*"
        self.children = [n for n in nodes]
        self.name = name
        self.leaf = False

        self.__set_serializable__()

    # def __getstate__(self):
    #     return {"id": self.id,
    #             "label": self.label,
    #             "children": self.children,
    #             "name": self.name,
    #             "leaf": self.leaf}

    def addChild(self, node):
        self.children.append(node)

    def size(self):
        return 1 + sum(map(lambda c: c.size(), self.children))

    def Prune(self):
        for c in self.children:
            c.Prune()

        while True:
            pruneNeeded = any(map(lambda c: type(c) == ProductNode, self.children))

            if not pruneNeeded:
                return

            newChildren = []
            for c in self.children:
                if type(c) == ProductNode:
                    for gc in c.children:
                        newChildren.append(gc)
                else:
                    newChildren.append(c)
            self.children = newChildren

    def validate(self):
        assert len(self.children) > 1, "not enough children in Product Node"

        for c in self.children:
            c.validate()

        self.scope = set()

        sum_features = 0
        for child in self.children:
            sum_features += len(child.scope)
            self.scope = self.scope | child.scope

        self.complete = all(map(lambda c: c.complete, self.children))
        self.consistent = len(self.scope) == sum_features

        assert self.consistent, "not consistent SPN"
        assert self.complete, "not complete SPN"

    def initTFData(self, X):
        self.X = X

    def __initLocalTfvars(self):
        self.validate()

        self.childrenprob = tf.stack([c.value for c in self.children], axis=1)
        # self.value = tf.reduce_prod(self.childrenprob, 1) #in prob space
        self.value = tf.reduce_sum(self.childrenprob, 1)  # in log space

    def initTf(self):
        self.validate()

        if hasattr(self, "X"):
            self.value = tf.log(self.X)
            return

        for c in self.children:
            c.initTf()

        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        for c in self.children:
            c.initTFSharedData(X, cache)

    def initMap(self, X, query=[]):
        self.validate()
        for c in self.children:
            c.initMap(X, query)

        with tf.name_scope(self.name):
            self.map = tf.reduce_sum(tf.stack([c.map for c in self.children], axis=0), 0)
            self.__initLocalTfvars()

    def marginalizeOut(self, marginals=None):
        self.validate()
        newChildren = [c if c.leaf else c.marginalizeOut(marginals) for c in self.children if (
            not c.leaf) or (c.featureIdx not in marginals)]

        newChildren = [c for c in newChildren if c]

        if len(newChildren) == 1:
            return newChildren[0]

        if len(newChildren) == 0:
            return None

        return ProductNode(self.name, *newChildren)

    def tftopy(self):
        for c in self.children:
            c.tftopy()

    def toEquation(self, evidence=None, fmt="python"):
        return "(" + " * ".join(map(lambda child:  child.toEquation(evidence, fmt), self.children)) + ")"


    def eval(self, data):
        return np.sum([c.eval(data) for c in self.children], axis=0)

    def mpe_eval(self, data):

        mpe_children_evals = [c.mpe_eval(data) for c in self.children]
        children_mpe_log_probs = np.array([c_lp for c_lp, _c_r in mpe_children_evals])
        children_mpe_res = np.array([c_r for _c_lp, c_r in mpe_children_evals])

        mpe_log_probs = np.sum(children_mpe_log_probs, axis=0)
        mpe_res = np.nansum(children_mpe_res, axis=0)

        #############################################################
        # print('child evals', mpe_children_evals)
        # # print(self.children)
        # print('child mpe log', children_mpe_log_probs)
        # assert mpe_log_probs[0] == mpe_log_probs[9], (mpe_log_probs[0], mpe_log_probs[9])
        # assert mpe_log_probs[3] == mpe_log_probs[4], (mpe_log_probs[3], mpe_log_probs[4])
        #############################################################

        return mpe_log_probs, mpe_res

    def sample(self, evidence_data, rand_gen=None):

        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        n_samples = evidence_data.shape[0]
        n_features = evidence_data.shape[1]
        n_children = len(self.children)

        sample_children_evals = [c.sample(evidence_data, rand_gen) for c in self.children]
        children_log_probs = np.array([c_lp for c_lp, _c_r in sample_children_evals])
        children_samples = np.array([c_r for _c_lp, c_r in sample_children_evals])

        sample_log_probs = np.sum(children_log_probs, axis=0)
        samples = np.nansum(children_samples, axis=0)

        return sample_log_probs, samples

    # def mpe_eval_scope(self, data):

    #     mpe_children_evals = [c.mpe_eval(data) for c in self.children]
    #     children_mpe_log_probs = np.array([c_lp for c_lp, _c_r in mpe_children_evals])
    #     children_mpe_res = np.array([c_r for _c_lp, c_r in mpe_children_evals])

    #     mpe_log_probs = np.sum(children_mpe_log_probs, axis=0)
    #     # mpe_res = np.nansum(children_mpe_res, axis=0)

    #     return mpe_log_probs, mpe_res

    def __repr__(self):
        print('prod name', self.name)
        print([c.name for c in self.children])
        pd = ", ".join(map(lambda c: c.name, self.children))
        result = "%s ProductNode(%s){\n\t" % (self.name, pd)
        result = result + "".join(map(lambda c: str(c), self.children)).replace("\n", "\n\t")
        return result + "}\n"


class SumNode(Node):

    def __init__(self, name, weights=None, *nodes):
        Node.__init__(self)
        self.label = "+"
        self.children = [n for n in nodes]
        self.name = name

        self.leaf = False

        self.weights = [w for w in weights] if weights else []
        self.log_weights = None

        self.log_weights = np.log(self.weights).reshape(len(self.children), 1) if weights else None

        self.__set_serializable__()

    # def __getstate__(self):
    #     return {"id": self.id,
    #             "label": self.label,
    #             "children": self.children,
    #             "name": self.name,
    #             "leaf": self.leaf,
    #             "weights": self.weights,
    #             "log_weights": self.log_weights}

    def addChild(self, weight, node):
        self.children.append(node)
        self.weights.append(weight)
        self.log_weights = np.log(self.weights).reshape(len(self.children), 1)

    def size(self):
        return 1 + sum(map(lambda c: c.size(), self.children))

    def Prune(self):
        for c in self.children:
            c.Prune()

        while True:
            pruneNeeded = any(map(lambda c: type(c) == SumNode, self.children))

            if not pruneNeeded:
                return

            newChildren = []
            newWeights = []
            for ci in range(len(self.children)):
                c = self.children[ci]
                if type(c) == SumNode:
                    for gci in range(len(c.children)):
                        gc = c.children[gci]
                        newChildren.append(gc)
                        newWeights.append(self.weights[ci] * c.weights[gci])
                else:
                    newChildren.append(c)
                    newWeights.append(self.weights[ci])
            self.children = newChildren
            self.weights = newWeights
            self.log_weights = np.log(self.weights).reshape(len(self.children), 1)

    def validate(self):
        assert len(self.children) > 1, "not enough children in Sum Node"

        for c in self.children:
            c.validate()

        self.scope = set(self.children[0].scope)
        self.complete = True
        self.consistent = all(map(lambda c: c.consistent, self.children))
        assert self.consistent, "not consistent SPN"

        for child in self.children[1:]:
            if self.scope != child.scope:
                self.complete = False
                break
        assert self.complete, "not complete SPN"

    def __initLocalTfvars(self):
        self.validate()
        self.tfweights = tf.nn.softmax(GetCreateVar(
            np.log(self.weights / np.max(self.weights)), scope="SPN/", context=self.name, name="weights"))
        self.childrenprob = tf.exp(tf.stack([c.value for c in self.children], axis=1))
        self.edgeprobs = tf.multiply(self.childrenprob, self.tfweights)
        self.value = tf.log(tf.reduce_sum(self.edgeprobs, 1))

    def initTf(self):
        self.validate()
        for c in self.children:
            c.initTf()

        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        for c in self.children:
            c.initTFSharedData(X, cache)

    def initMap(self, X, query=[]):
        self.validate()

        for c in self.children:
            c.initMap(X, query)

        with tf.name_scope(self.name):
            self.__initLocalTfvars()

            self.cmaps = tf.stack([c.map for c in self.children], axis=1)
            self.maxprobs = tf.stack(
                [tf.range(0, tf.shape(X)[0]), tf.cast(tf.arg_max(self.edgeprobs, 1), tf.int32)], axis=1)
            self.map = tf.gather_nd(self.cmaps, self.maxprobs)

    def marginalizeOut(self, marginals=None):
        self.validate()
        newChildren = [c if c.leaf else c.marginalizeOut(marginals) for c in self.children if (
            not c.leaf) or (c.featureIdx not in marginals)]

        newChildren = [c for c in newChildren if c]

        if len(newChildren) == 0:
            return None

        return SumNode(self.name, self.weights, *newChildren)

    def tftopy(self):
        self.weights = self.tfweights.eval()
        self.log_weights = np.log(self.weights).reshape(len(self.children), 1)

        for c in self.children:
            c.tftopy()

    def toEquation(self, evidence=None, fmt="python"):
        weights = self.weights

        weights = weights / np.sum(weights)

        # weights[-1] = 1.0 - sum(weights[0:-1])

        return "(" + " + ".join(map(lambda i: str(weights[i]) + "*(" + self.children[i].toEquation(evidence, fmt) + ")", range(len(self.children)))) + ")"


    def eval(self, data):
        llchildren = np.array([c.eval(data) for c in self.children])

        return scipy.misc.logsumexp(llchildren + self.log_weights, axis=0)

    def mpe_eval(self, data):

        n_samples = data.shape[0]
        n_features = data.shape[1]
        n_children = len(self.children)
        mpe_children_evals = [c.mpe_eval(data) for c in self.children]
        children_mpe_log_probs = np.array([c_lp for c_lp, _c_r in mpe_children_evals])
        children_mpe_res = np.array([c_r for _c_lp, c_r in mpe_children_evals])

        w_children_evals = self.log_weights + children_mpe_log_probs
        mpe_children_ids = np.argmax(w_children_evals, axis=0)
        # print('MPE children log prob {}'.format(self.id),
        #       self.log_weights[:, None] + children_mpe_log_probs)
        # print('MPE children id {}'.format(self.id), mpe_children_ids)
        mpe_children_ids_mask = mpe_children_ids[None, :] == np.arange(len(self.children))[:, None]
        # mpe_children_ids_mask = np.ix_(mpe_children_ids, np.arange(len(self.children)))
        # print(mpe_children_ids_mask)
        # print(mpe_children_ids.shape)
        # print(children_mpe_log_probs.shape)
        # print(self.log_weights.shape, children_mpe_log_probs.shape, w_children_evals.shape,
        #       mpe_children_ids.shape, mpe_children_ids_mask.shape)
        # print(mpe_children_ids_mask)
        # print(w_children_evals)
        # mpe_log_probs = children_mpe_log_probs[mpe_children_ids_mask]

        mpe_log_probs = w_children_evals[np.arange(n_children)[mpe_children_ids],
                                         np.arange(n_samples)]
        # mpe_log_probs = w_children_evals[mpe_children_ids_mask]

        # print(mpe_log_probs.shape)
        # mpe_children_ids_tensor = np.ix_(mpe_children_ids,
        #                                  np.arange(len(self.children)),
        #                                  np.arange(n_features))
        # mpe_children_ids_mask = mpe_children_ids[None, :] ==
        # np.arange(len(self.children))[:, None]
        # print(children_mpe_res.shape)
        # print(children_mpe_res)
        # mask_list = [mpe_children_ids_mask] * n_features
        # print(mask_list)
        # mpe_children_ids_tensor = np.stack(mask_list, axis=2)
        # print(mpe_children_ids_tensor)
        mpe_res = np.zeros((n_samples, n_features))
        mpe_res[:] = np.nan
        # print(children_mpe_res[mpe_children_ids_mask])
        for s in range(n_samples):
            max_child_id = mpe_children_ids[s]
            mpe_res[s] = children_mpe_res[max_child_id, s]
        # mpe_res = children_mpe_res[mpe_children_ids_tensor]

        #########################################
        # print(mpe_log_probs)
        # assert mpe_log_probs[0] == mpe_log_probs[9], (mpe_log_probs[0], mpe_log_probs[9])
        # assert mpe_log_probs[3] == mpe_log_probs[4], (mpe_log_probs[3], mpe_log_probs[4])
        ########################################

        return mpe_log_probs, mpe_res

    def sample(self, evidence_data, rand_gen=None):

        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        n_samples = evidence_data.shape[0]
        n_features = evidence_data.shape[1]
        n_children = len(self.children)

        sample_children_evals = [c.sample(evidence_data, rand_gen) for c in self.children]
        children_log_probs = np.array([c_lp for c_lp, _c_r in sample_children_evals])
        children_samples = np.array([c_r for _c_lp, c_r in sample_children_evals])

        #
        # now selecting one child branch proportionally to the probabilites
        w_children_log_probs = self.log_weights + children_log_probs
        children_probs = np.exp(w_children_log_probs)
        children_probs = children_probs / children_probs.sum(axis=0)
        rand_child_branches = np.array([rand_gen.choice(np.arange(n_children), p=children_probs[:, i])
                                        for i in range(n_samples)])

        # children_ids_mask = rand_child_branches[None, :] == np.arange(n_children)[:, None]

        # sample_log_probs = children_log_probs[children_ids_mask]
        sample_log_probs = w_children_log_probs[np.arange(n_children)[rand_child_branches],
                                                np.arange(n_samples)]
        # print(sample_log_probs.shape)
        samples = np.zeros((n_samples, n_features))
        samples[:] = np.nan

        for s in range(n_samples):
            sampled_child_id = rand_child_branches[s]
            samples[s] = children_samples[sampled_child_id, s]

        return sample_log_probs, samples

    def soft_sample(self, evidence_data, rand_gen=None):

        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        n_samples = evidence_data.shape[0]
        n_features = evidence_data.shape[1]
        n_children = len(self.children)

        sample_children_evals = [c.sample(evidence_data, rand_gen) for c in self.children]
        children_log_probs = np.array([c_lp for c_lp, _c_r in sample_children_evals])
        children_samples = np.array([c_r for _c_lp, c_r in sample_children_evals])

        w_children_log_probs = self.log_weights + children_log_probs

        children_probs = np.exp(w_children_log_probs)
        children_probs = children_probs / children_probs.sum(axis=0)
        sample_log_probs = scipy.misc.logsumexp(w_children_log_probs, axis=0)

        samples = children_probs[:, :, None] * children_samples
        samples = np.sum(samples, axis=0)

        return sample_log_probs, samples

    def __repr__(self):
        w = self.weights
        sumw = ", ".join(map(
            lambda i: "%s*%s" % (w[i], self.children[i].name if self.children[i] else "None"), range(len(self.children))))
        result = "%s SumNode(%s){\n\t" % (self.name, sumw)
        result = result + "".join(map(lambda c: str(c), self.children)).replace("\n", "\n\t")
        return result + "}\n"


class BernoulliNode(Node):

    families = set(['bernoulli'])

    def __init__(self, name, featureIdx, featureName, p):
        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.p = p
        self.setLabel()

        self.__set_serializable__()

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def __initLocalTfvars(self):
        self.tfp = tf.minimum(tf.maximum(tf.sigmoid(GetCreateVar(
            logit(self.p), scope="SPN/", context=self.name, name="p")), 0.000000000000001), 0.999999999999990)
        self.dist = distributions.Bernoulli(probs=self.tfp)
        self.value = tf.log(tf.maximum(self.dist.prob(self.X), 0.000000000000001))

    def initTf(self):
        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        if self.featureIdx in cache:
            result = cache[self.featureIdx]
        else:
            result = X[:, self.featureIdx]
            cache[self.featureIdx] = result
        self.X = result

    def initTFData(self, X):
        self.X = X

    def initMap(self, X, query=[]):
        self.initTFData(X[:, self.featureIdx])

        mv = np.zeros(X.get_shape()[1])

        with tf.name_scope(self.name):
            self.__initLocalTfvars()
            if self.featureIdx in query:
                mapval = round(self.p)
                mv[self.featureIdx] = mapval
                # print(mapval, self.p, mv)
                self.value = tf.log(self.dist.prob(tf.zeros_like(X[:, self.featureIdx]) + mapval))
            self.map = tf.ones_like(X) * mv

    def tftopy(self):
        self.p = self.tfp.eval()
        self.setLabel()

    def toEquation(self, evidence=None, fmt="python"):
        if evidence and self.featureIdx in evidence:
            return "%.30f" % (bernoullipmf(evidence[self.featureIdx], self.p))

        return "bernoullipmf(x_{featureIdx}_, {p})".format(**{'p': self.p, 'featureIdx': self.featureIdx})

    def setLabel(self):
        self.label = "B({featureName}|ρ={p:.3f})".format(
            **{'p': self.p, 'featureName': self.featureName})

    def eval(self, data):
        return nplogbernoullipmf_fast(data[:, self.featureIdx], self.p)

    def mpe_eval(self, data):
        obs = data[:, self.featureIdx]

        query_ids = np.isnan(obs)

        mpe_res = np.zeros(data.shape)
        mpe_res[:] = np.nan
        mpe_log_probs = np.zeros(obs.shape)
        mpe_log_probs[:] = LOG_ZERO

        mpe_log_probs[query_ids] = np.log(self.p) if self.p > 0.5 else np.log(1-self.p)
        mpe_res[query_ids, self.featureIdx] = 1 if self.p > 0.5 else 0

        mpe_log_probs[~query_ids] = self.eval(data[~query_ids])
        mpe_res[~query_ids, self.featureIdx] = obs[~query_ids]

        assert np.isnan(mpe_res[:, self.featureIdx]).sum() == 0

        return mpe_log_probs, mpe_res

    def sample(self, evidence_data, rand_gen=None):
        """
        Sampling leverages
        """

        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        obs = evidence_data[:, self.featureIdx]

        # sampling from the mixture of uniform box prior or from the actual estimated density

        query_ids = np.isnan(obs)
        n_samples = sum(query_ids)

        samples = np.zeros(evidence_data.shape)
        samples[:] = np.nan
        sample_log_probs = np.zeros(obs.shape)

        new_samples = np.array([1 if i > self.p else 0 for i in rand_gen.uniform(low=1.0, high=1.0, size=n_samples)])
        samples[query_ids, self.featureIdx] = new_samples.reshape(new_samples.shape[0])

        #
        # for observed values
        # FIXME: this is inefficient, we are slicing data again and again
        if (~query_ids).sum() > 0:
            sample_log_probs[~query_ids] = self.eval(evidence_data[~query_ids])
            samples[~query_ids, self.featureIdx] = obs[~query_ids]

        return sample_log_probs, samples

    def __repr__(self):
        return "%s %s\n" % (self.name, self.label)


class PoissonNode(Node):

    families = set(['poisson'])

    def __init__(self, name, featureIdx, featureName, mean):
        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.mean = mean if mean > 0 else 0.000000000000001

        self.setLabel()

        self.__set_serializable__()

    # def __getstate__(self):
    #     return {"label": self.label, "children": self.children, "name": self.name, "leaf": self.leaf, "featureName": self.featureName,
    #             "featureIdx": self.featureIdx, "scope": self.scope, "consistent": self.consistent, "complete": self.complete, "mean": self.mean}

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def __initLocalTfvars(self):
        self.tfmean = tf.maximum(
            GetCreateVar(self.mean, scope="SPN/", context=self.name, name="mean"), 0.000000000000001)
        self.dist = distributions.Poisson(rate=self.tfmean)
        self.value = tf.log(tf.maximum(self.dist.prob(self.X), 0.000000000000001))

    def initTf(self):
        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        if self.featureIdx in cache:
            result = cache[self.featureIdx]
        else:
            result = X[:, self.featureIdx]
            cache[self.featureIdx] = result
        self.X = result

    def initTFData(self, X):
        self.X = X

    def initMap(self, X, query=[]):
        self.initTFData(X[:, self.featureIdx])

        mv = np.zeros(X.get_shape()[1])

        with tf.name_scope(self.name):
            self.__initLocalTfvars()
            if self.featureIdx in query:
                mapval = round(self.mean)
                mv[self.featureIdx] = mapval
                self.value = tf.log(self.dist.pmf(tf.zeros_like(X[:, self.featureIdx]) + mapval))
            self.map = tf.ones_like(X) * mv

    def tftopy(self):
        self.mean = self.tfmean.eval()
        self.setLabel()

    def toEquation(self, evidence=None, fmt="python"):
        if evidence and self.featureIdx in evidence:
            return "%.30f" % (poissonpmf(evidence[self.featureIdx], self.mean))

        if fmt == "python":
            return "poissonpmf(x_{featureIdx}_, {mean})".format(**{'mean': self.mean, 'featureIdx': self.featureName})
        elif fmt == "mathematica":
            return "PDF[PoissonDistribution[{mean}], x_{featureIdx}_]".format(**{'mean': self.mean, 'featureIdx': self.featureIdx})

    def setLabel(self):
        self.label = "P({featureName}|λ={mean:.3f})".format(
            **{'mean': self.mean, 'featureName': self.featureName})

    def eval(self, data):
        return nplogpoissonpmf(data[:, self.featureIdx], self.mean)

    def __repr__(self):
        return "%s %s\n" % (self.name, self.label)


class GaussianNode(Node):

    families = set(['gaussian', 'normal'])

    def __init__(self, name, featureIdx, featureName, mean, stdev):
        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.mean = mean

        if np.isclose(stdev, 0.0):
            stdev = 0.00001

        if stdev < 0.4:
            stdev = 0.4

        self.stdev = stdev
        self.variance = self.stdev * self.stdev
        self.setLabel()

        self.__set_serializable__()

    # def __getstate__(self):
    #     return {"label": self.label, "children": self.children, "name": self.name, "leaf": self.leaf, "featureName": self.featureName,
    #             "featureIdx": self.featureIdx, "scope": self.scope, "consistent": self.consistent, "complete": self.complete,
    #             "mean": self.mean, "stdev": self.stdev, "variance": self.variance}

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def __initLocalTfvars(self):
        self.tfmean = GetCreateVar(self.mean, scope="SPN/", context=self.name, name="mean")
        self.tfstdev = tf.maximum(
            GetCreateVar(self.stdev, scope="SPN/", context=self.name, name="stdev"), 0.4)

        self.dist = distributions.Normal(loc=self.tfmean, scale=self.tfstdev)
        self.value = tf.log(tf.maximum(self.dist.prob(self.X), 0.000000000000001))

    def initTf(self):
        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        if self.featureIdx in cache:
            result = cache[self.featureIdx]
        else:
            result = X[:, self.featureIdx]
            cache[self.featureIdx] = result
        self.X = result

    def initTFData(self, X):
        self.X = X

    def initMap(self, X, query=[]):
        self.initTFData(X[:, self.featureIdx])

        mv = np.zeros(X.get_shape()[1])

        with tf.name_scope(self.name):
            self.__initLocalTfvars()
            if self.featureIdx in query:
                mv[self.featureIdx] = self.mean
                self.value = tf.log(
                    self.dist.prob(tf.zeros_like(X[:, self.featureIdx]) + self.mean))
            self.map = tf.ones_like(X) * mv

    def tftopy(self):
        self.mean = self.tfmean.eval()
        self.stdev = self.tfstdev.eval()
        self.setLabel()

    def toEquation(self, evidence=None, fmt="python"):
        if evidence and self.featureIdx in evidence:
            return "%.30f" % (gaussianpdf(evidence[self.featureIdx], self.mean, self.variance))

        return "gaussianpdf(x_{featureIdx}_, {mean}, {variance})".format(**{'mean': self.mean, 'featureIdx': self.featureIdx, 'variance': self.variance})

    def setLabel(self):
        self.label = "G({featureName}|μ={mean:.3f},σ={stdev:.3f})".format(
            **{'mean': self.mean, 'stdev': self.stdev, 'featureName': self.featureName})

    def eval(self, data):

        vals = scipy.stats.norm.pdf(data[:, self.featureIdx], self.mean, self.stdev)

        return np.log(vals)

    def mpe_eval(self, data):
        obs = data[:, self.featureIdx]

        query_ids = np.isnan(obs)

        mpe_res = np.zeros(data.shape)
        mpe_res[:] = np.nan
        mpe_log_probs = np.zeros(obs.shape)
        mpe_log_probs[:] = LOG_ZERO

        _data = obs
        _data[query_ids] = self.mean

        # TODO: Problem with the mean and the data array
        mpe_log_probs[query_ids] = np.log(self.eval(data[query_ids]))
        mpe_res[query_ids, self.featureIdx] = _data[query_ids]

        mpe_log_probs[~query_ids] = self.eval(data[~query_ids])
        mpe_res[~query_ids, self.featureIdx] = obs[~query_ids]

        assert np.isnan(mpe_res[:, self.featureIdx]).sum() == 0

        return mpe_log_probs, mpe_res

    def sample(self, evidence_data, rand_gen=None):
        """
        Sampling leverages
        """

        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        obs = evidence_data[:, self.featureIdx]

        # sampling from the mixture of uniform box prior or from the actual estimated density

        query_ids = np.isnan(obs)
        n_samples = sum(query_ids)

        samples = np.zeros(evidence_data.shape)
        samples[:] = np.nan
        sample_log_probs = np.zeros(obs.shape)

        new_samples = rand_gen.normal(loc=self.mean, scale=self.stdev, size=n_samples)
        samples[query_ids, self.featureIdx] = new_samples.reshape(new_samples.shape[0])

        # for observed values
        # FIXME: this is inefficient, we are slicing data again and again
        if (~query_ids).sum() > 0:
            sample_log_probs[~query_ids] = self.eval(evidence_data[~query_ids])
            samples[~query_ids, self.featureIdx] = obs[~query_ids]

        return sample_log_probs, samples

    def __repr__(self):
        return "%s %s\n" % (self.name, self.label)


class PiecewiseLinearPDFNodeOld(Node):

    families = set(['piecewise-old'])

    def __init__(self, name,
                 featureIdx, featureName,
                 domain, x_range, y_range,
                 prior_weight, prior_density):

        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.prior_density = prior_density
        self.log_prior_density = np.log(prior_density)
        self.prior_weight = prior_weight
        self.log_prior_weight = np.log(prior_weight)

        self.domain = domain
        self.x_range = x_range
        self.y_range = y_range
        self.setLabel()

        self.__set_serializable__()

    # def __getstate__(self):
    #     return {"label": self.label, "children": self.children, "name": self.name, "leaf": self.leaf, "featureName": self.featureName,
    #             "featureIdx": self.featureIdx, "scope": self.scope, "consistent": self.consistent, "complete": self.complete,
    #             "domain": self.domain, "x_range": self.x_range, "y_range": self.y_range}

    def setLabel(self):
        # self.label =
        # "PWL({featureName}|domain={domain},x_range={x_range},y_range={y_range})".format(**{'featureName':
        # self.featureName,
        self.label = "PWL-old({featureIdx}({featureName}))".format(**{
            'featureIdx': self.featureIdx,
            'featureName': self.featureName})

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    @jit
    def eval(self, data):
        """
        1. get the interval in which input falls
        2. interpolate
        both could be achieved with np.interp
        """
        # values outside the provided interval are assumed to have zero mass
        obs = data[:, self.featureIdx]

        lt = obs < (self.domain[0] - EPSILON)
        mt = obs > (self.domain[-1] + EPSILON)

        outside_domain = np.logical_or(lt, mt)
        assert outside_domain.sum() == 0

        result = np.zeros(obs.shape)
        result[:] = LOG_ZERO

        ivalues = np.interp(x=obs, xp=self.x_range, fp=self.y_range)
        ividx = ivalues > 0
        result[ividx] = np.log(ivalues[ividx])

        result[np.logical_or(lt, mt)] = LOG_ZERO

        # return np.logaddexp(self.log_prior_weight + self.log_prior_density,
        #                     np.log(1 - self.prior_weight) + result)

        # if (result == LOG_ZERO).sum() > 0:
        #     print('\n\n\n\nPIECEWISE gets zero ll\n')
        #     print(self)
        #     print(np.nonzero(result == LOG_ZERO))

        return result

    def mpe_eval(self, data):
        """
        Computing MPE (log) probabilities and assignment in one pass
        data is a query data matrix of (n_queries, n_features)
        where each query instance can contain NaN where a query RV is

        Return: an array of (n_queries) log probabilies and a matrix
        of the same size of data where the column corresponding to the score of the leaf node
        containes the computed MPE assignments for the queries
        """
        obs = data[:, self.featureIdx]

        query_ids = np.isnan(obs)

        mpe_res = np.zeros(data.shape)
        mpe_res[:] = np.nan
        mpe_log_probs = np.zeros(obs.shape)
        mpe_log_probs[:] = LOG_ZERO

        _x = np.argmax(self.y_range)
        mpe_y = self.y_range[_x]
        mpe_x = self.x_range[_x]

        #
        # is it a queried value?
        mpe_log_probs[query_ids] = np.log(mpe_y)
        mpe_res[query_ids, self.featureIdx] = mpe_x

        #
        # for observed values
        # FIXME: this is inefficient, we are slicing data again and again
        mpe_log_probs[~query_ids] = self.eval(data[~query_ids])
        mpe_res[~query_ids, self.featureIdx] = obs[~query_ids]

        # # à#######################
        # if self.featureIdx >= 10:
        #     assert query_ids.sum() == 0
        # else:
        #     assert query_ids.sum() == data.shape[0]

        # assert mpe_res[0, self.featureIdx] == mpe_res[
        #     9, self.featureIdx], (mpe_res[0, self.featureIdx], mpe_res[9, self.featureIdx])
        # assert mpe_res[3, self.featureIdx] == mpe_res[
        #     4, self.featureIdx], (mpe_res[3, self.featureIdx], mpe_res[4, self.featureIdx])
        # assert mpe_log_probs[0] == mpe_log_probs[9], (mpe_log_probs[0], mpe_log_probs[9])
        # assert mpe_log_probs[3] == mpe_log_probs[4], (mpe_log_probs[3], mpe_log_probs[4])
        # #######################################

        assert np.isnan(mpe_res[:, self.featureIdx]).sum() == 0

        return mpe_log_probs, mpe_res

    def sample(self, evidence_data, rand_gen=None):

        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        obs = evidence_data[:, self.featureIdx]

        query_ids = np.isnan(obs)
        n_samples = sum(query_ids)

        samples = np.zeros(evidence_data.shape)
        samples[:] = np.nan
        sample_log_probs = np.zeros(obs.shape)
        #
        # NOTE: if the value is not observed, we have to marginalize in the forward pass
        # sample_log_probs[:] = 0.

        #
        # for unobserved values
        samples[query_ids, self.featureIdx] = two_staged_sampling_piecewise_linear(self.x_range,
                                                                                   self.y_range,
                                                                                   n_samples=n_samples,
                                                                                   sampling='rejection',
                                                                                   rand_gen=rand_gen)

        #
        # for observed values
        # FIXME: this is inefficient, we are slicing data again and again
        sample_log_probs[~query_ids] = self.eval(evidence_data[~query_ids])
        samples[~query_ids, self.featureIdx] = obs[~query_ids]

        return sample_log_probs, samples

    def __repr__(self):
        return "%s %s %s %s\n" % (self.name, self.label, self.x_range, self.y_range,)


class PiecewiseLinearPDFNode(Node):

    families = set(['piecewise'])

    def __init__(self, name,
                 featureIdx, featureName,
                 domain, x_range, y_range,
                 prior_weight, prior_density,
                 bin_repr_points):

        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.prior_density = prior_density
        self.log_prior_density = np.log(prior_density)
        self.prior_weight = prior_weight
        self.log_prior_weight = np.log(prior_weight)

        self.domain = domain
        self.x_range = x_range
        self.y_range = y_range

        self.bin_repr_points = bin_repr_points
        self.setLabel()

        self.__set_serializable__()

    def setLabel(self):
        # self.label =
        # "PWL({featureName}|domain={domain},x_range={x_range},y_range={y_range})".format(**{'featureName':
        # self.featureName,
        self.label = "PWL({featureIdx}({featureName}))".format(**{
            'featureIdx': self.featureIdx,
            'featureName': self.featureName})

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    @jit
    def eval(self, data):
        """
        1. get the interval in which input falls
        2. interpolate
        both could be achieved with np.interp
        """
        # values outside the provided interval are assumed to have zero mass
        obs = data[:, self.featureIdx]

        lt = obs < (self.domain[0] - EPSILON)
        mt = obs > (self.domain[-1] + EPSILON)

        outside_domain = np.logical_or(lt, mt)
        assert outside_domain.sum() == 0, (obs[lt], obs[mt], self.domain)

        result = np.zeros(obs.shape)
        result[:] = LOG_ZERO

        ivalues = np.interp(x=obs, xp=self.x_range, fp=self.y_range)
        ividx = ivalues > 0
        result[ividx] = np.log(ivalues[ividx])

        # print(self.featureIdx, self.name, result)
        # result[np.logical_or(lt, mt)] = LOG_ZERO

        return np.logaddexp(self.log_prior_weight + self.log_prior_density,
                            np.log(1 - self.prior_weight) + result)
        # return np.log(self.prior_weight * self.prior_density +
        #               (1 - self.prior_weight) * np.exp(result))

        # if (result == LOG_ZERO).sum() > 0:
        #     print('\n\n\n\nPIECEWISE gets zero ll\n')
        #     print(self)
        #     print(np.nonzero(result == LOG_ZERO))

        return result

    def mpe_eval(self, data):
        """
        Computing MPE (log) probabilities and assignment in one pass
        data is a query data matrix of (n_queries, n_features)
        where each query instance can contain NaN where a query RV is

        Return: an array of (n_queries) log probabilies and a matrix
        of the same size of data where the column corresponding to the score of the leaf node
        containes the computed MPE assignments for the queries
        """
        obs = data[:, self.featureIdx]

        query_ids = np.isnan(obs)

        mpe_res = np.zeros(data.shape)
        mpe_res[:] = np.nan
        mpe_log_probs = np.zeros(obs.shape)
        mpe_log_probs[:] = LOG_ZERO

        log_unif = self.log_prior_weight + self.log_prior_density

        _x = np.argmax(self.y_range)

        mpe_y = self.y_range[_x]
        mpe_x = self.x_range[_x]

        #
        # is it a queried value?
        log_mpe_y = np.log(mpe_y)
        if log_mpe_y > log_unif:
            mpe_log_probs[query_ids] = log_mpe_y
        else:
            mpe_log_probs[query_ids] = log_unif
        #
        # always putting the best state according to the original density
        mpe_res[query_ids, self.featureIdx] = mpe_x

        #
        # for observed values
        # FIXME: this is inefficient, we are slicing data again and again
        mpe_log_probs[~query_ids] = self.eval(data[~query_ids])
        mpe_res[~query_ids, self.featureIdx] = obs[~query_ids]

        # # à#######################
        # if self.featureIdx >= 10:
        #     assert query_ids.sum() == 0
        # else:
        #     assert query_ids.sum() == data.shape[0]

        # assert mpe_res[0, self.featureIdx] == mpe_res[
        #     9, self.featureIdx], (mpe_res[0, self.featureIdx], mpe_res[9, self.featureIdx])
        # assert mpe_res[3, self.featureIdx] == mpe_res[
        #     4, self.featureIdx], (mpe_res[3, self.featureIdx], mpe_res[4, self.featureIdx])
        # assert mpe_log_probs[0] == mpe_log_probs[9], (mpe_log_probs[0], mpe_log_probs[9])
        # assert mpe_log_probs[3] == mpe_log_probs[4], (mpe_log_probs[3], mpe_log_probs[4])
        # #######################################

        assert np.isnan(mpe_res[:, self.featureIdx]).sum() == 0

        return mpe_log_probs, mpe_res

    def sample(self, evidence_data, rand_gen=None):

        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        obs = evidence_data[:, self.featureIdx]

        query_ids = np.isnan(obs)
        n_samples = sum(query_ids)

        samples = np.zeros(evidence_data.shape)
        samples[:] = np.nan
        sample_log_probs = np.zeros(obs.shape)
        #
        # NOTE: if the value is not observed, we have to marginalize in the forward pass
        # sample_log_probs[:] = 0.

        #
        # for unobserved values
        samples[query_ids, self.featureIdx] = two_staged_sampling_piecewise_linear(self.x_range,
                                                                                   self.y_range,
                                                                                   n_samples=n_samples,
                                                                                   sampling='rejection',
                                                                                   rand_gen=rand_gen)

        #
        # for observed values
        # FIXME: this is inefficient, we are slicing data again and again
        sample_log_probs[~query_ids] = self.eval(evidence_data[~query_ids])
        samples[~query_ids, self.featureIdx] = obs[~query_ids]

        return sample_log_probs, samples

    def __repr__(self):
        return "%s %s %s %s %s %s\n" % (self.name, self.label, self.x_range, self.y_range,
                                        self.prior_weight, self.prior_density)


class IsotonicUnimodalPDFNode(PiecewiseLinearPDFNode):

    families = set(['isotonic'])

    def __init__(self, name,
                 featureIdx, featureName,
                 domain, x_range, y_range,
                 prior_weight, prior_density,
                 bin_repr_points):

        PiecewiseLinearPDFNode.__init__(self, name,
                                        featureIdx, featureName,
                                        domain, x_range, y_range,
                                        prior_weight, prior_density,
                                        bin_repr_points)

        self.__set_serializable__()

    def setLabel(self):

        self.label = "ISO({featureIdx}({featureName}))".format(**{
            'featureIdx': self.featureIdx,
            'featureName': self.featureName})


class HistNode(Node):

    families = set(['histogram'])

    def __init__(self, name, featureIdx, featureName, breaks, densities, prior_density,
                 prior_weight,
                 bin_repr_points):
        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.breaks = breaks
        self.densities = densities
        self.prior_density = prior_density
        self.prior_weight = prior_weight
        self.log_prior_density = np.log(prior_density)
        self.log_prior_weight = np.log(prior_weight)

        self.bin_repr_points = bin_repr_points
        self.setLabel()

        self.__set_serializable__()

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def setLabel(self):
        self.label = "Hist({featureName}|pw={prior_weight},pd={prior_density})".format(
            **{'featureName': self.featureName, 'prior_weight': self.prior_weight, 'prior_density': self.prior_density})

    def eval(self, data):
        data = data[:, self.featureIdx]

        probs = (self.prior_weight) * np.ones_like(data) * self.prior_density

        import bisect
        for i, x in enumerate(data):
            outsideLeft = bisect.bisect(self.breaks, x) == 0
            outsideRight = bisect.bisect_left(self.breaks, x) == len(self.breaks)
            outside = outsideLeft or outsideRight

            if outside:
                continue

            density = self.densities[bisect.bisect_left(self.breaks, x) - 1]

            probs[i] += (1.0 - self.prior_weight) * density

        return np.log(probs)

    def mpe_eval(self, data):
        obs = data[:, self.featureIdx]

        query_ids = np.isnan(obs)

        mpe_res = np.zeros(data.shape)
        mpe_res[:] = np.nan
        mpe_log_probs = np.zeros(obs.shape)
        mpe_log_probs[:] = LOG_ZERO

        log_unif = self.log_prior_weight + self.log_prior_density

        _x = np.argmax(self.densities)

        mpe_y = self.densities[_x]
        # mpe_x = self.breaks[_x] + (self.breaks[_x + 1] - self.breaks[_x]) / 2
        mpe_x = self.bin_repr_points[_x]

        #
        # is it a queried value?
        log_mpe_y = np.log(mpe_y)
        if log_mpe_y > log_unif:
            mpe_log_probs[query_ids] = log_mpe_y
        else:
            mpe_log_probs[query_ids] = log_unif
        #
        # always putting the best state according to the original density
        mpe_res[query_ids, self.featureIdx] = mpe_x

        #
        # for observed values
        # FIXME: this is inefficient, we are slicing data again and again
        mpe_log_probs[~query_ids] = self.eval(data[~query_ids])
        mpe_res[~query_ids, self.featureIdx] = obs[~query_ids]

        # # à#######################
        # if self.featureIdx >= 10:
        #     assert query_ids.sum() == 0
        # else:
        #     assert query_ids.sum() == data.shape[0]

        # assert mpe_res[0, self.featureIdx] == mpe_res[
        #     9, self.featureIdx], (mpe_res[0, self.featureIdx], mpe_res[9, self.featureIdx])
        # assert mpe_res[3, self.featureIdx] == mpe_res[
        #     4, self.featureIdx], (mpe_res[3, self.featureIdx], mpe_res[4, self.featureIdx])
        # assert mpe_log_probs[0] == mpe_log_probs[9], (mpe_log_probs[0], mpe_log_probs[9])
        # assert mpe_log_probs[3] == mpe_log_probs[4], (mpe_log_probs[3], mpe_log_probs[4])
        # #######################################

        assert np.isnan(mpe_res[:, self.featureIdx]).sum() == 0

        return mpe_log_probs, mpe_res

    def __repr__(self):
        return "%s %s %s %s\n" % (self.name, self.label, np.array2string(self.breaks, precision=10, separator=','), np.array2string(self.densities, precision=10, separator=','),)


class KernelDensityEstimatorNode(Node):

    families = set(['kde'])

    def __init__(self,
                 name,
                 featureIdx, featureName,
                 data, domain,
                 kernel='gaussian',
                 bandwidth=0.2,
                 metric='euclidean',
                 prior_density=None,
                 prior_weight=0.01):

        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.prior_density = prior_density
        self.log_prior_density = np.log(prior_density)
        self.prior_weight = prior_weight
        self.log_prior_weight = np.log(prior_weight)

        self.domain = domain
        self._kernel = kernel
        self._bandwidth = bandwidth
        self._metric = metric
        #
        # creating an estimator with parameters
        self._kde = KernelDensity(bandwidth=self._bandwidth,
                                  algorithm='auto',
                                  kernel=self._kernel,
                                  metric=self._metric,
                                  atol=0,
                                  rtol=0,
                                  breadth_first=True,
                                  leaf_size=40,
                                  metric_params=None)

        #
        # fit on the data
        if data.ndim == 1:
            data = data.reshape(data.shape[0], -1)

        fit_start_t = perf_counter()
        self._kde.fit(data)
        fit_end_t = perf_counter()
        logging.debug('\t\tfit {} kde estimator for leaf {}'.format(self._kernel,
                                                                    self.featureIdx))

        self.setLabel()

        self.__set_serializable__()

    def setLabel(self):

        self.label = "KDELeaf({featureIdx}({featureName}))".format(**{
            'featureIdx': self.featureIdx,
            'featureName': self.featureName})

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def eval(self, data):
        """
        Evaluate by scoring the log likelihood via sklearn
        """

        obs = data[:, self.featureIdx]

        lt = obs < (self.domain[0] - EPSILON)
        mt = obs > (self.domain[-1] + EPSILON)
        outside_domain = np.logical_or(lt, mt)
        assert outside_domain.sum() == 0, outside_domain

        obs = obs.reshape(obs.shape[0], -1)
        # print(obs.shape)

        kde_eval = self._kde.score_samples(obs)

        # return np.logaddexp(self.log_prior_weight + self.log_prior_density,
        #                     np.log(1 - self.prior_weight) + kde_eval)

        # print('prior weight', self.prior_weight)
        # print('log prior weight', self.log_prior_weight)
        # print('prior density', self.prior_density)
        # print('log prior density', self.log_prior_density)
        return np.log(self.prior_weight * self.prior_density +
                      (1 - self.prior_weight) * np.exp(kde_eval))
        # return kde_eval

    @staticmethod
    def check_eval(kde_node, n_features):

        def kde_check_eval(x):

            obs = np.array([[0 for i in range(n_features)]])
            obs[:, kde_node.featureIdx] = x
            # print(obs)
            # print(obs.shape)
            ll = None
            if x < kde_node.domain.min() or x > kde_node.domain.max():
                ll = kde_node._kde.score_samples([[x]])
            else:
                # ll = kde_node._kde.score_samples([[x]])
                ll = kde_node.eval(obs)
            return np.exp(ll[0])

        from scipy.integrate import quad

        # result, prec = quad(kde_check_eval, kde_node.domain.min(), kde_node.domain.max())
        result, prec = quad(kde_check_eval, -np.inf, np.inf)
        if abs(result - 1.0) > 0.000001:
            raise ValueError('Unnormalized leaf', result, prec)

    def mpe_eval(self, data):
        """
        How to do MPE with KDE? we can use the domain, and for each point in the chosen resolution,
        brute force compute the MPE (linear in the number of points of the domain)
        """

        raise NotImplementedError('MPE not implemented for KDE')

    def sample(self, evidence_data, rand_gen=None):
        """
        Sampling leverages
        """

        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        obs = evidence_data[:, self.featureIdx]

        #
        # sampling from the mixture of uniform box prior or from the actual estimated density

        query_ids = np.isnan(obs)
        n_samples = sum(query_ids)

        samples = np.zeros(evidence_data.shape)
        samples[:] = np.nan
        sample_log_probs = np.zeros(obs.shape)
        #
        # NOTE: if the value is not observed, we have to marginalize in the forward pass
        # sample_log_probs[:] = 0.

        #
        # for unobserved values
        kde_samples = self._kde.sample(n_samples=n_samples, random_state=rand_gen)
        samples[query_ids, self.featureIdx] = kde_samples.reshape(kde_samples.shape[0])

        #
        # for observed values
        # FIXME: this is inefficient, we are slicing data again and again
        if (~query_ids).sum() > 0:
            sample_log_probs[~query_ids] = self.eval(evidence_data[~query_ids])
            samples[~query_ids, self.featureIdx] = obs[~query_ids]

        return sample_log_probs, samples

    def __repr__(self):
        return "%s %s %s\n" % (self.name, self.label, self._kde)


class GaussianKDENode(KernelDensityEstimatorNode):

    families = set(['gkde'])


class GammaNode(Node):

    families = set(['gamma'])

    def __init__(self, name, featureIdx, featureName, concentration, rate):
        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.concentration = concentration
        self.rate = rate
        self.scale = 1 / rate  # do we need this other parametrization?

        self.setLabel()

        self.__set_serializable__()

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def __initLocalTfvars(self):
        self.tfconcentration = GetCreateVar(
            self.concentration, scope="SPN/", context=self.name, name="concentration")
        self.tfrate = GetCreateVar(self.rate, scope="SPN/", context=self.name, name="rate")

        self.dist = distributions.Gamma(concentration=self.tfconcentration, rate=self.tfrate)
        self.value = tf.log(tf.maximum(self.dist.prob(self.X), 0.000000000000001))

    def initTf(self):
        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        if self.featureIdx in cache:
            result = cache[self.featureIdx]
        else:
            result = X[:, self.featureIdx]
            cache[self.featureIdx] = result
        self.X = result

    def initTFData(self, X):
        self.X = X

    def initMap(self, X, query=[]):

        raise NotImplementedError('MAP not implemented for gamma, yet')
        # self.initTFData(X[:, self.featureIdx])

        # mv = np.zeros(X.get_shape()[1])

        # with tf.name_scope(self.name):
        #     self.__initLocalTfvars()
        #     if self.featureIdx in query:
        #         mv[self.featureIdx] = self.mean
        #         self.value = tf.log(
        #             self.dist.pdf(tf.zeros_like(X[:, self.featureIdx]) + self.mean))
        #     self.map = tf.ones_like(X) * mv

    def tftopy(self):
        self.concentration = self.tfconcentration.eval()
        self.rate = self.tfrate.eval()
        self.scale = 1 / self.rate
        self.setLabel()

    def toEquation(self, evidence=None, fmt="python"):
        if evidence and self.featureIdx in evidence:
            return "%.30f" % (gammapdf(evidence[self.featureIdx], self.concentration, self.rate))

        return "gammapdf(x_{featureIdx}_, {loc}, {rate})".format(**{'loc': self.concentration,
                                                                    'featureIdx': self.featureIdx,
                                                                    'rate': self.rate})

    def setLabel(self):
        self.label = "Gamma({featureName}|μ={loc:.3f},σ={rate:.3f})".format(
            **{'loc': self.concentration,
               'rate': self.rate,
               'featureName': self.featureName})

    def eval(self, data):

        # vals = scipy.stats.gamma.pdf(data[:, self.featureIdx],
        #                              a=self.concentration,
        #                              scale=self.scale)
        vals = gammapdf(data[:, self.featureIdx],
                        concentration=self.concetration,
                        rate=self.rate)

        return np.log(vals)

    def __repr__(self):
        return "%s %s\n" % (self.name, self.label)


class CategoricalNode(Node):

    families = set(['categorical'])

    def __init__(self, name, featureIdx, featureName, probs):
        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.probs = probs

        assert np.isclose(np.sum(probs), 1.0), (self.probs, np.sum(probs))
        self.values = len(probs)

        self.setLabel()

        self.__set_serializable__()

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def __initLocalTfvars(self):
        self.tfprobs = GetCreateVar(
            self.probs, scope="SPN/", context=self.name, name="probs")

        self.dist = distributions.Categorical(probs=self.probs)

    def initTf(self):
        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        if self.featureIdx in cache:
            result = cache[self.featureIdx]
        else:
            result = X[:, self.featureIdx]
            cache[self.featureIdx] = result
        self.X = result

    def initTFData(self, X):
        self.X = X

    def initMap(self, X, query=[]):

        raise NotImplementedError('MAP not implemented for categorical, yet')

    def tftopy(self):
        self.probs = self.tfprobs.eval()
        self.setLabel()

    def toEquation(self, evidence=None, fmt="python"):
        if evidence and self.featureIdx in evidence:
            return "%.30f" % (self.probs[evidence[self.featureIdx].astype(int)])

        return "categoricalpmf(x_{featureIdx}_, {probs})".format(**{'probs': self.probs,
                                                                    'featureIdx': self.featureIdx})

    def setLabel(self):
        self.label = "Categorical({featureName}|probs={probs})".format(
            **{'probs': self.probs,
               'featureName': self.featureName})

    def eval(self, data):

        vals = self.probs[data[:, self.featureIdx].astype(int)]
        return np.log(vals)

    def __repr__(self):
        return "%s %s\n" % (self.name, self.label)


class BetaNode(Node):

    families = set(['beta'])

    def __init__(self, name, featureIdx, featureName, alpha, beta):
        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.alpha = alpha
        self.beta = beta

        self.setLabel()

        self.__set_serializable__()

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def __initLocalTfvars(self):
        self.tfalpha = GetCreateVar(
            self.alpha, scope="SPN/", context=self.name, name="alpha")
        self.tfbeta = GetCreateVar(self.beta, scope="SPN/", context=self.name, name="beta")

        self.dist = distributions.Beta(concentration1=self.tfalpha, concentration0=self.beta)
        self.value = tf.log(tf.maximum(self.dist.pdf(self.X), 0.000000000000001))

    def initTf(self):
        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        if self.featureIdx in cache:
            result = cache[self.featureIdx]
        else:
            result = X[:, self.featureIdx]
            cache[self.featureIdx] = result
        self.X = result

    def initTFData(self, X):
        self.X = X

    def initMap(self, X, query=[]):

        raise NotImplementedError('MAP not implemented for beta, yet')
        # self.initTFData(X[:, self.featureIdx])

        # mv = np.zeros(X.get_shape()[1])

        # with tf.name_scope(self.name):
        #     self.__initLocalTfvars()
        #     if self.featureIdx in query:
        #         mv[self.featureIdx] = self.mean
        #         self.value = tf.log(
        #             self.dist.pdf(tf.zeros_like(X[:, self.featureIdx]) + self.mean))
        #     self.map = tf.ones_like(X) * mv

    def tftopy(self):
        self.alpha = self.tfalpha.eval()
        self.beta = self.tfbeta.eval()
        self.setLabel()

    def toEquation(self, evidence=None, fmt="python"):
        if evidence and self.featureIdx in evidence:
            return "%.30f" % (betapdf(evidence[self.featureIdx], self.alpha, self.beta))

        return "betapdf(x_{featureIdx}_, {alpha}, {beta})".format(**{'alpha': self.alpha,
                                                                     'featureIdx': self.featureIdx,
                                                                     'beta': self.beta})

    def setLabel(self):
        self.label = "Beta({featureName}|alpha={alpha:.3f},beta={beta:.3f})".format(
            **{'alpha': self.alpha,
               'beta': self.beta,
               'featureName': self.featureName})

    def eval(self, data):

        vals = betapdf(data[:, self.featureIdx],
                       alpha=self.alpha,
                       besta=self.beta)

        return np.log(vals)

    def __repr__(self):
        return "%s %s\n" % (self.name, self.label)


class DiscreteNode(Node):

    families = set(['discrete'])

    def __init__(self, name, featureIdx, featureName, value):
        Node.__init__(self)
        self.featureName = featureName
        self.children = []
        self.name = name
        self.featureIdx = featureIdx
        self.scope = set([featureIdx])
        self.complete = True
        self.consistent = True
        self.leaf = True

        self.value = value

        self.setLabel()

        self.__set_serializable__()

    def size(self):
        return 1

    def Prune(self):
        pass

    def validate(self):
        pass

    def __initLocalTfvars(self):
        ninf_tensor = tf.cast(tf.fill(dims=tf.shape(self.X), value=-1e20), self.X.dtype)
        self.value = tf.where(tf.equal(self.X, self.value), tf.zeros_like(self.X), ninf_tensor)

    def initTf(self):
        with tf.name_scope(self.name):
            self.__initLocalTfvars()

    def initTFSharedData(self, X, cache={}):
        if self.featureIdx in cache:
            result = cache[self.featureIdx]
        else:
            result = X[:, self.featureIdx]
            cache[self.featureIdx] = result
        self.X = result

    def initTFData(self, X):
        self.X = X

    def initMap(self, X, query=[]):
        raise NotImplementedError('MAP not implemented for discrete, yet')

    def tftopy(self):
        self.setLabel()

    def toEquation(self, evidence=None, fmt="python"):
        if evidence and self.featureIdx in evidence:
            return '1' if evidence[self.featureIdx] == self.value else '0'

        return "discrete(x_{featureIdx}_, {value})".format(**{'value': self.value,
                                                              'featureIdx': self.featureIdx})

    def setLabel(self):
        self.label = "Discrete({featureName}|value={value:.3f})".format(
            **{'value': self.value,
               'featureName': self.featureName})

    def eval(self, data):
        raise NotImplementedError()

    def __repr__(self):
        return "%s %s\n" % (self.name, self.label)


def Perplexity(jointNode, bow):
    words = np.sum(bow)
    Px = jointNode.value
    ll = tf.reduce_sum(tf.log(Px))
    pwb = ll / words
    return (pwb, np.exp2(-pwb), words, ll)


def JointCost(jointNode):
    # minimize negative log likelihood
    Px = jointNode.value
    costf = -tf.reduce_sum(Px)
    return costf


def DiscriminativeCost(jointNode, marginalNode):

    Pxy = jointNode.value
    Px = marginalNode.value

    # minimize negative log likelihood
    costf = tf.reduce_sum(Px - Pxy)
    return costf
