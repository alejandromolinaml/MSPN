from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import numpy
from numpy.testing import assert_array_equal

import os
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

from tfspn.tfspn import PiecewiseLinearPDFNode
from tfspn.piecewise import continuous_data_to_piecewise_linear_pdf
from tfspn.piecewise import isotonic_unimodal_regression_R
from tfspn.piecewise import merge_piecewise_linear_observations

from tfspn.piecewise import piecewise_linear_approximation
from tfspn.piecewise import domain_to_bins
from tfspn.piecewise import compute_histogram
from tfspn.piecewise import averaging_histograms
from tfspn.piecewise import histogram_counts_to_frequences
from tfspn.piecewise import piecewise_linear_from_histogram
from tfspn.piecewise import merge_piecewise_linear_observations
from tfspn.piecewise import isotonic_unimodal_regression_R
from tfspn.piecewise import estimate_bins_and_domains_from_data

RAND_STATE = 1337


def test_bernoulli_as_piecewise_linear():
    perc = 0.31
    n_obs = 200
    data = numpy.random.binomial(n=1, p=perc, size=n_obs).astype(int)

    #
    # MLE estimate bernoulli's theta
    theta = numpy.sum(data) / n_obs
    print('MLE theta {}'.format(theta))

    #
    # test for smoothing
    laplace_smoothing = 0.0

    #
    # determining domain and bins
    domain = numpy.array([0, 1])
    center = True
    step = 1
    bins = [domain_to_bins(domain, step=step, center=center)]
    print('\tfrom domain to bins {} -> {}'.format(domain, bins[0]))

    x, y = piecewise_linear_approximation(data,
                                          bins,
                                          center=center,
                                          bin_width=step,
                                          alpha=laplace_smoothing,
                                          n_bootstraps=None,
                                          average_bootstraps=True,
                                          remove_duplicates=True,
                                          isotonic=True,
                                          rand_gen=None)

    print(x, y)
    assert numpy.allclose(x[1:-1], domain)

    bernoulli_node = PiecewiseLinearPDFNode(0, domain=domain, x_range=x, y_range=y)

    #
    # check for the one
    input_obs = numpy.array([1.])
    bernoulli_node.eval(input_obs)
    log_theta = bernoulli_node.log_val
    exp_theta = numpy.exp(log_theta)
    print('\t bernoulli node log/theta for 1: {} {}'.format(log_theta, exp_theta))
    assert numpy.allclose(exp_theta, theta)

    #
    # check for the zero
    input_obs = numpy.array([0.])
    bernoulli_node.eval(input_obs)
    log_theta = bernoulli_node.log_val
    exp_theta = numpy.exp(log_theta)
    print('\t bernoulli node log/theta for 0: {} {}'.format(log_theta, exp_theta))
    assert numpy.allclose(exp_theta, 1. - theta)

    #
    # check for a value out of the domain (zero probability expected)
    input_obs = numpy.array([-0.5])
    bernoulli_node.eval(input_obs)
    log_theta = bernoulli_node.log_val
    exp_theta = numpy.exp(log_theta)
    print('\t bernoulli node log/theta for out of domain: {} {}'.format(log_theta, exp_theta))
    assert numpy.allclose(exp_theta, 0)

    #
    # check for a value out of the domain (zero probability expected)
    input_obs = numpy.array([1.01])
    bernoulli_node.eval(input_obs)
    log_theta = bernoulli_node.log_val
    exp_theta = numpy.exp(log_theta)
    print('\t bernoulli node log/theta for out of domain: {} {}'.format(log_theta, exp_theta))
    assert numpy.allclose(exp_theta, 0)


def test_merge_piecewise_linear_observations():

    X = [numpy.array([0, 0.5, 1, 1.5, 2]),
         numpy.array([0, 1, 2, 3, 4, 5]),
         numpy.array([0.5, 0.7, 0.9])]

    Y = [numpy.arange(len(X[i])) for i in range(len(X))]

    x, y = merge_piecewise_linear_observations(X, Y, remove_duplicates=False)
    print(x, y)

    x, y = merge_piecewise_linear_observations(X, Y, remove_duplicates=True)
    print(x, y)


def test_estimate_bins_and_domains_from_data():

    rand_gen = numpy.random.RandomState(RAND_STATE)

    n_instances = 10

    data = []
    type_labels = []

    #
    # some bernoullis to treat like categoricals or discrete
    theta = 0.28
    n_bernoulli = 3
    b_data_labels = ['categorical' for i in range(n_bernoulli)]
    type_labels += b_data_labels
    b_data = rand_gen.binomial(n=1, p=theta, size=(n_instances, n_bernoulli))
    data.append(b_data)

    #
    # some continuous data
    mean = 10
    std = 2.0
    n_normals = 3
    n_data_labels = ['continuous' for i in range(n_normals)]
    type_labels += n_data_labels
    n_data = rand_gen.normal(size=(n_instances, n_normals), loc=mean, scale=std)
    data.append(n_data)

    #
    # some integer data
    n_discretes = 3
    MAX_DISCRETE = 100
    d_data_labels = ['discrete' for i in range(n_normals)]
    type_labels += d_data_labels
    mins = [rand_gen.choice(MAX_DISCRETE) for i in range(n_discretes)]
    maxes = [mins[1] + MAX_DISCRETE for i in range(n_discretes)]
    d_data = numpy.zeros((n_instances, n_discretes))
    for i in range(n_instances):
        for j in range(n_discretes):
            d_data[i, j] = rand_gen.choice(numpy.arange(mins[j], maxes[j]))
    data.append(d_data)

    data = numpy.concatenate(data, axis=1)
    print(data, type_labels)
    bins, domains = estimate_bins_and_domains_from_data(data=data,
                                                        type_labels=type_labels,
                                                        binning_method=["auto"],
                                                        center_bins=True,
                                                        float_type=numpy.float64,
                                                        int_type=numpy.int32)

    j = 0
    for i, (b, d, l) in enumerate(zip(bins, domains, type_labels)):
        print('Feature type: {}'.format(l))
        print('\tBinnings: {}'.format(b))
        print('\tDomain: {}'.format(d))

        if l == 'discrete':

            d_m = d - 0.5
            d_m = [dd for dd in d_m]
            d_m += [d_m[-1] + 1]
            assert_array_equal(b[0], d_m)
            #j += 1


def test_piecewise_mpe_eval():

    perc = 0.31
    n_obs = 200
    data = numpy.random.binomial(n=1, p=perc, size=n_obs).astype(int)

    #
    # MLE estimate bernoulli's theta
    theta = numpy.sum(data) / n_obs
    print('MLE theta {}'.format(theta))

    #
    # test for smoothing
    laplace_smoothing = 0.0

    #
    # determining domain and bins
    domain = numpy.array([0, 1])
    center = True
    step = 1
    bins = [domain_to_bins(domain, step=step, center=center)]
    print('\tfrom domain to bins {} -> {}'.format(domain, bins[0]))

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

    bernoulli_node = PiecewiseLinearPDFNode("node0",
                                            0,
                                            'f0',
                                            domain=domain, x_range=x, y_range=y)

    #
    # check for the one
    input_obs = numpy.array([[1., numpy.nan, 3.], [numpy.nan, 0., 1.], [numpy.nan, 1.0, 0.]])
    probs, res = bernoulli_node.mpe_eval(input_obs)

    f_id = bernoulli_node.featureIdx
    n_features = input_obs.shape[1]
    other_feature_ids = numpy.array([i for i in range(f_id)] +
                                    [i for i in range(f_id + 1, n_features)])
    assert numpy.all(numpy.isnan(res[:, other_feature_ids]))

    print('\t bernoulli node log prob and mpe assignment: {} {}'.format(probs, res))

    bernoulli_node_2 = PiecewiseLinearPDFNode("node1",
                                              1,
                                              'f1',
                                              domain=domain, x_range=x, y_range=y)
    probs_2, res_2 = bernoulli_node_2.mpe_eval(input_obs)

    f_id_2 = bernoulli_node_2.featureIdx
    other_feature_ids = numpy.array([i for i in range(f_id_2)] +
                                    [i for i in range(f_id_2 + 1, n_features)])
    assert numpy.all(numpy.isnan(res_2[:, other_feature_ids]))

    print('\t bernoulli node 2 log prob and mpe assignment: {} {}'.format(probs_2, res_2))


# def test_product_node_eval():

    # from tfspn.tfspn import BernoulliNode
    # from tfspn.tfspn import BernoulliNodeGood
    # from tfspn.tfspn import ProductNode

    # rand_gen = numpy.random.RandomState(RAND_STATE)

    # n_children = 10
    # node_names = ['bernoulli-leaf-{}'.format(i) for i in range(n_children)]
    # feature_id = 0
    # feature_name = 'f5'
    # n_params = 5

    # thetas = rand_gen.rand(n_params, 2)
    # thetas = thetas / thetas.sum(axis=1, keepdims=True)
    # print("Random generated Bernoulli parameters:\n{}".format(thetas))

    # p = 0.666
    # n_samples = 10
    # input_data = rand_gen.binomial(n=1, p=p, size=n_samples)
    # print(input_data)

    # for t in thetas:
    #     children = [BernoulliNode(name=node_names[k],
    #                               featureIdx=feature_id,
    #                               featureName=feature_name,
    #                               p=t[1])
    #                 for k in range(n_children)]

    #     #
    #     # creating sum node, testing add_child
    #     prod_node = ProductNode(name='prod-node')

    #     for c in children:
    #         prod_node.addChild(c)

    #     #
    #     # evaluating children before parents
    #     for obs in input_data:
    #         for c in children:
    #             c.eval(obs)
    #         children_log_probs = [c.log_val for c in children]

    #         prod_node.eval()
    #         log_prob = prod_node.log_val

    #         expected_log_val = numpy.sum(children_log_probs)
    #         assert_almost_equal(log_prob, expected_log_val)


if __name__ == '__main__':
    test_piecewise_mpe_eval()
