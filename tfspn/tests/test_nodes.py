"""
Testing node evaluation in
   - linked representation
   - tensorflow
"""
import numpy
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

import tensorflow as tf
from tfspn.SPN import SPN, Splitting
# from tfspn.tfspn import SumNode, ProductNode, PoissonNode, GaussianNode, BernoulliNodeGood

RAND_STATE = 1337


def tf_eval_node_test(node, leaves, input_data, expected_values):

    with tf.device("/cpu:0"):
        tf.reset_default_graph()

        with tf.name_scope('input'):
            X = tf.placeholder(tf.float64, [None, 2], name="x")

        with tf.name_scope('leaves'):
            for l in leaves:
                l.initTFSharedData(X)

        with tf.name_scope('node') as scope:
            node.initTf()

    with tf.Session() as sess:
        # variables need to be initialized before we can use them
        sess.run(tf.global_variables_initializer())

        # this computes the probabilities at the root
        log_probs = node.value.eval({X: input_data})
        print(node)
        print(log_probs)


def test_tf_sum_node_over_indicator_leaves():

    #
    # building the network
    #

    s = SumNode('sum-node-test')
    c1 = BernoulliNodeGood('id-var-1', featureIdx=0, featureName='var1', p=1.0)
    c2 = BernoulliNodeGood('id-var-2', featureIdx=0, featureName='var1', p=0.0)
    w1 = 0.5
    w2 = 0.5
    s.addChild(w1, c1)
    s.addChild(w2, c1)

    print(s)
    data = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # tf_eval_node_test(s, leaves=[c1, c2], input_data=data, expected_values=None)
    tf_eval_node_test(c1, leaves=[c1], input_data=data, expected_values=None)

#
# FIXME: these tests are useless since we switched to the recursive version


def test_bernoulli_node_eval():

    #
    # creating a bernoulli node
    #
    # FIXME: which one is the good one?
    from tfspn.tfspn import BernoulliNode
    from tfspn.tfspn import BernoulliNodeGood

    rand_gen = numpy.random.RandomState(RAND_STATE)

    node_name = 'bernoulli-leaf'
    feature_id = 0
    feature_name = 'f5'
    n_params = 100

    thetas = rand_gen.rand(n_params, 2)
    thetas = thetas / thetas.sum(axis=1, keepdims=True)
    print("Random generated Bernoulli parameters:\n{}".format(thetas))

    #
    # input data whose probability has to be computed
    # NOTE: this is not a single instance
    p = 0.666
    n_samples = 10
    input_data = rand_gen.binomial(n=1, p=p, size=n_samples)
    print(input_data)

    for t in thetas:
        print('Considering theta: {}'.format(t))

        bn = BernoulliNode(name=node_name,
                           featureIdx=feature_id,
                           featureName=feature_name,
                           p=t[1])

        for obs in input_data:
            bn.eval(obs)
            log_prob = bn.log_val
            assert_almost_equal(log_prob, numpy.log(t[obs]))

#
# FIXME: these tests are useless since we switched to the recursive version


def test_piecewise_node_eval():

    #
    # creating a piecewise linear node modeling a Bernoulli
    #
    # FIXME: which one is the good one?
    from tfspn.tfspn import PiecewiseLinearPDFNode
    from tfspn.piecewise import domain_to_bins

    rand_gen = numpy.random.RandomState(RAND_STATE)

    node_name = 'piecewise-leaf'
    feature_id = 0
    feature_name = 'f5'
    n_params = 100

    thetas = rand_gen.rand(n_params, 2)
    thetas = thetas / thetas.sum(axis=1, keepdims=True)
    print("Random generated Bernoulli parameters:\n{}".format(thetas))

    #
    # input data whose probability has to be computed
    # NOTE: this is not a single instance
    p = 0.666
    n_samples = 10
    input_data = rand_gen.binomial(n=1, p=p, size=n_samples)
    print(input_data)

    bernoulli_domain = numpy.array([0, 1])
    bernoulli_range = numpy.array([-0.5] + [dv for dv in bernoulli_domain] + [1.5])
    print(bernoulli_range)

    for t in thetas:

        print('Considering theta: {}'.format(t))

        t_vals = numpy.array([0.0] + [tv for tv in t] + [0.0])
        print('Considering theta for piecewise: {}'.format(t_vals))

        pwln = PiecewiseLinearPDFNode(name=node_name,
                                      var=feature_id,
                                      domain=bernoulli_domain,
                                      x_range=bernoulli_range,
                                      y_range=t_vals,
                                      featureName=feature_name)

        for obs in input_data:
            pwln.eval(obs)
            log_prob = pwln.log_val
            assert_almost_equal(log_prob, numpy.log(t[obs]))

#
# FIXME: these tests are useless since we switched to the recursive version


def test_sum_node_eval():

    from scipy.misc import logsumexp

    from tfspn.tfspn import BernoulliNode
    from tfspn.tfspn import BernoulliNodeGood
    from tfspn.tfspn import SumNode
    #
    # creating some input bernoulli nodes
    #
    # FIXME: which one is the good one?

    rand_gen = numpy.random.RandomState(RAND_STATE)

    n_children = 10
    node_names = ['bernoulli-leaf-{}'.format(i) for i in range(n_children)]
    feature_id = 0
    feature_name = 'f5'
    n_params = 5

    thetas = rand_gen.rand(n_params, 2)
    thetas = thetas / thetas.sum(axis=1, keepdims=True)
    print("Random generated Bernoulli parameters:\n{}".format(thetas))

    p = 0.666
    n_samples = 10
    input_data = rand_gen.binomial(n=1, p=p, size=n_samples)
    print(input_data)

    for t in thetas:
        children = [BernoulliNode(name=node_names[k],
                                  featureIdx=feature_id,
                                  featureName=feature_name,
                                  p=t[1])
                    for k in range(n_children)]

        #
        # NOTE: these are weights that must sum to 1
        weights = rand_gen.rand(n_children)
        weights = weights / weights.sum()
        log_weights = numpy.log(weights)

        #
        # creating sum node, testing add_child
        sum_node = SumNode(name='sum-node')

        for c, w in zip(children, weights):
            sum_node.addChild(w, c)

        assert_array_almost_equal(log_weights, sum_node.log_weights)

        #
        # evaluating children before parents
        for obs in input_data:
            for c in children:
                c.eval(obs)
            children_log_probs = [c.log_val for c in children]

            sum_node.eval()
            log_prob = sum_node.log_val

            expected_log_val = logsumexp(children_log_probs + log_weights)

            assert_almost_equal(log_prob, expected_log_val)

#
# NOTE: these tests are useless since we switched to the recursive version


def test_product_node_eval():

    from tfspn.tfspn import BernoulliNode
    from tfspn.tfspn import BernoulliNodeGood
    from tfspn.tfspn import ProductNode

    rand_gen = numpy.random.RandomState(RAND_STATE)

    n_children = 10
    node_names = ['bernoulli-leaf-{}'.format(i) for i in range(n_children)]
    feature_id = 0
    feature_name = 'f5'
    n_params = 5

    thetas = rand_gen.rand(n_params, 2)
    thetas = thetas / thetas.sum(axis=1, keepdims=True)
    print("Random generated Bernoulli parameters:\n{}".format(thetas))

    p = 0.666
    n_samples = 10
    input_data = rand_gen.binomial(n=1, p=p, size=n_samples)
    print(input_data)

    for t in thetas:
        children = [BernoulliNode(name=node_names[k],
                                  featureIdx=feature_id,
                                  featureName=feature_name,
                                  p=t[1])
                    for k in range(n_children)]

        #
        # creating sum node, testing add_child
        prod_node = ProductNode(name='prod-node')

        for c in children:
            prod_node.addChild(c)

        #
        # evaluating children before parents
        for obs in input_data:
            for c in children:
                c.eval(obs)
            children_log_probs = [c.log_val for c in children]

            prod_node.eval()
            log_prob = prod_node.log_val

            expected_log_val = numpy.sum(children_log_probs)
            assert_almost_equal(log_prob, expected_log_val)


def test_product_node_mpe_eval():

    from tfspn.tfspn import ProductNode
    from tfspn.tfspn import PiecewiseLinearPDFNode

    from tfspn.piecewise import domain_to_bins
    from tfspn.piecewise import piecewise_linear_approximation

    rand_gen = numpy.random.RandomState(RAND_STATE)

    n_features = 4
    n_children = n_features
    node_names = ['pwnode-leaf-{}'.format(i) for i in range(n_children)]
    feature_id = 0
    feature_name = 'f5'
    n_params = 1

    thetas = rand_gen.rand(n_params, 2)
    thetas = thetas / thetas.sum(axis=1, keepdims=True)
    print("Random generated Bernoulli parameters:\n{}".format(thetas))

    p = 0.666
    n_samples = 10
    input_data = rand_gen.binomial(n=1, p=p, size=(n_samples, n_features))
    print(input_data)

    rand_x = rand_gen.choice(n_samples, size=(n_samples // 2))
    rand_y = rand_gen.choice(n_features, size=(n_samples // 2))
    input_data = input_data.astype(numpy.float64)
    input_data[rand_x, rand_y] = numpy.nan
    print(input_data)

    domain = numpy.array([0, 1])
    center = True
    step = 1
    bins = [domain_to_bins(domain, step=step, center=center)]
    laplace_smoothing = 1
    print('\tfrom domain to bins {} -> {}'.format(domain, bins[0]))

    for t in thetas:

        #
        # creating a product node, testing add_child
        prod_node = ProductNode(name='prod-node')

        for k in range(n_children):
            data = numpy.random.binomial(n=1, p=t[1], size=n_samples).astype(int)
            x, y = piecewise_linear_approximation(data,
                                                  bins,
                                                  family='categorical',
                                                  bin_width=step,
                                                  alpha=laplace_smoothing,
                                                  n_bootstraps=None,
                                                  average_bootstraps=True,
                                                  remove_duplicates=True,
                                                  isotonic=True,
                                                  rand_gen=None)

            print(x, y)
            assert numpy.allclose(x[1:-1], domain)

            fid = rand_gen.choice(n_features)
            c = PiecewiseLinearPDFNode(node_names[k],
                                       featureIdx=k,
                                       featureName='f{}'.format(fid),
                                       domain=domain, x_range=x, y_range=y)
            print(c)

            prod_node.addChild(c)

        #
        prob, res = prod_node.mpe_eval(input_data)
        print('mpe log probs {}'.format(prob))
        print('mpe assignment {}'.format(res))

        for lp, s in zip(prob, res):
            assert_almost_equal(lp, prod_node.eval(s.reshape(-1, s.shape[0]))[0])


def test_sum_node_mpe_eval():

    from tfspn.tfspn import ProductNode
    from tfspn.tfspn import SumNode
    from tfspn.tfspn import PiecewiseLinearPDFNode

    from tfspn.piecewise import domain_to_bins
    from tfspn.piecewise import piecewise_linear_approximation

    rand_gen = numpy.random.RandomState(RAND_STATE)

    n_features = 4
    n_params = 1
    thetas = rand_gen.rand(n_params, 2)
    thetas = thetas / thetas.sum(axis=1, keepdims=True)
    print("Random generated Bernoulli parameters:\n{}".format(thetas))

    p = 0.666
    n_samples = 10
    input_data = rand_gen.binomial(n=1, p=p, size=(n_samples, n_features))
    print(input_data)

    rand_x = rand_gen.choice(n_samples, size=(n_samples // 2))
    rand_y = rand_gen.choice(n_features, size=(n_samples // 2))
    input_data = input_data.astype(numpy.float64)
    input_data[rand_x, rand_y] = numpy.nan
    print(input_data)

    domain = numpy.array([0, 1])
    center = True
    step = 1
    bins = [domain_to_bins(domain, step=step, center=center)]
    laplace_smoothing = 1
    print('\tfrom domain to bins {} -> {}'.format(domain, bins[0]))

    for t in thetas:

        #
        # creating sum node over two product nodes
        prod_node_1 = ProductNode(name='prod-node-1')
        prod_node_2 = ProductNode(name='prod-node-2')

        data_11 = rand_gen.binomial(n=1, p=t[1], size=n_samples).astype(int)
        x_11, y_11 = piecewise_linear_approximation(data_11,
                                                    bins,
                                                    family='categorical',
                                                    bin_width=step,
                                                    alpha=laplace_smoothing,
                                                    n_bootstraps=None,
                                                    average_bootstraps=True,
                                                    remove_duplicates=True,
                                                    isotonic=True,
                                                    rand_gen=None)

        print(x_11, y_11)
        assert numpy.allclose(x_11[1:-1], domain)

        fid = 0
        c_11 = PiecewiseLinearPDFNode('11',
                                      featureIdx=fid,
                                      featureName='f{}'.format(fid),
                                      domain=domain, x_range=x_11, y_range=y_11)
        print(c_11)

        data_12 = rand_gen.binomial(n=1, p=t[0], size=n_samples).astype(int)
        x_12, y_12 = piecewise_linear_approximation(data_12,
                                                    bins,
                                                    family='categorical',
                                                    bin_width=step,
                                                    alpha=laplace_smoothing,
                                                    n_bootstraps=None,
                                                    average_bootstraps=True,
                                                    remove_duplicates=True,
                                                    isotonic=True,
                                                    rand_gen=None)

        print(x_12, y_12)
        assert numpy.allclose(x_12[1:-1], domain)

        fid = 1
        c_12 = PiecewiseLinearPDFNode('12',
                                      featureIdx=fid,
                                      featureName='f{}'.format(fid),
                                      domain=domain, x_range=x_12, y_range=y_12)
        print(c_12)

        prod_node_1.addChild(c_11)
        prod_node_1.addChild(c_12)

        data_21 = rand_gen.binomial(n=1, p=t[0], size=n_samples).astype(int)
        x_21, y_21 = piecewise_linear_approximation(data_21,
                                                    bins,
                                                    family='categorical',
                                                    bin_width=step,
                                                    alpha=laplace_smoothing,
                                                    n_bootstraps=None,
                                                    average_bootstraps=True,
                                                    remove_duplicates=True,
                                                    isotonic=True,
                                                    rand_gen=None)

        print(x_21, y_21)
        assert numpy.allclose(x_21[1:-1], domain)

        fid = 0
        c_21 = PiecewiseLinearPDFNode('21',
                                      featureIdx=fid,
                                      featureName='f{}'.format(fid),
                                      domain=domain, x_range=x_21, y_range=y_21)
        print(c_21)

        data_22 = rand_gen.binomial(n=1, p=t[1], size=n_samples).astype(int)
        x_22, y_22 = piecewise_linear_approximation(data_22,
                                                    bins,
                                                    family='categorical',
                                                    bin_width=step,
                                                    alpha=laplace_smoothing,
                                                    n_bootstraps=None,
                                                    average_bootstraps=True,
                                                    remove_duplicates=True,
                                                    isotonic=True,
                                                    rand_gen=None)

        print(x_22, y_22)
        assert numpy.allclose(x_22[1:-1], domain)

        fid = 1
        c_22 = PiecewiseLinearPDFNode('21',
                                      featureIdx=fid,
                                      featureName='f{}'.format(fid),
                                      domain=domain, x_range=x_22, y_range=y_22)
        print(c_22)

        prod_node_2.addChild(c_21)
        prod_node_2.addChild(c_22)

        prob, res = c_11.mpe_eval(input_data)
        print('c11 mpe log probs {}'.format(prob))
        print('c11 mpe assignment {}'.format(res))
        prob, res = c_12.mpe_eval(input_data)
        print('c12 mpe log probs {}'.format(prob))
        print('c12 mpe assignment {}'.format(res))

        prob, res = c_21.mpe_eval(input_data)
        print('c21 mpe log probs {}'.format(prob))
        print('c21 mpe assignment {}'.format(res))
        prob, res = c_22.mpe_eval(input_data)
        print('c22 mpe log probs {}'.format(prob))
        print('c22 mpe assignment {}'.format(res))

        prob, res = prod_node_1.mpe_eval(input_data)
        print('prod 1 mpe log probs {}'.format(prob))
        print('prod 1 mpe assignment {}'.format(res))

        prob, res = prod_node_2.mpe_eval(input_data)
        print('prod 2 mpe log probs {}'.format(prob))
        print('prod 2 mpe assignment {}'.format(res))

        sum_node = SumNode(name='sum-node')
        sum_node.addChild(0.5, prod_node_1)
        sum_node.addChild(0.5, prod_node_2)

        #
        prob, res = sum_node.mpe_eval(input_data)
        print('mpe log probs {}'.format(prob))
        print('mpe assignment {}'.format(res))

        # for lp, s in zip(prob, res):
        #     assert_almost_equal(lp, prod_node.eval(s.reshape(-1, s.shape[0]))[0])

if __name__ == '__main__':
    # test_product_node_mpe_eval()
    test_sum_node_mpe_eval()
