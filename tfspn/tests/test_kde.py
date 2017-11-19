from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import numpy


def test_gaussian_kde_leaf():
    from tfspn.tfspn import KernelDensityEstimatorNode
    # import matplotlib.pyplot as pyplot

    #
    # generate some continuous data, sampling from long tailed
    rand_gen = numpy.random.RandomState(1337)
    n_samples = 100
    mu, beta = 0, 0.1  # location and scale
    samples_g1 = rand_gen.gumbel(mu, beta, n_samples)
    mu, beta = 1.0, 0.5  # location and scale
    samples_g2 = rand_gen.gumbel(mu, beta, n_samples)
    samples = numpy.concatenate([samples_g1, samples_g2])

    # pyplot.hist(samples)
    # pyplot.show()

    samples = samples.reshape(samples.shape[0], -1)
    print(samples.shape)

    #
    # create node
    kernel = 'gaussian'
    bw = 0.2
    metric = 'euclidean'
    kde_node = KernelDensityEstimatorNode(name='kde-0',
                                          featureIdx=0,
                                          featureName='f0',
                                          data=samples,
                                          domain=None,
                                          kernel=kernel,
                                          bandwidth=bw,
                                          metric=metric)

    input_data = rand_gen.gumbel(mu, beta, size=(n_samples, 4))
    print(input_data.shape)
    lls = kde_node.eval(input_data)
    print('lls', lls)

    #
    # sampling from kde
    n_draws = 100
    data_ev = numpy.zeros((n_draws, 4))
    data_ev[:] = numpy.nan
    probs, samples = kde_node.sample(data_ev, rand_gen)
    print(probs)
    print(samples)

    def kde_eval(x):
        ll = kde_node.eval(numpy.array([[x]]))
        print(ll.shape)
        return numpy.exp(ll[0])

    # prior_weight = 0.01
    # prior_density = 1
    # def kde_mix_eval(x):
    #     ll = kde_node.eval(numpy.array([[x]]))
    #     # print(ll.shape)
    #     numpy.log(prior_weight * self.prior_density +
    #                   (1 - self.prior_weight) * np.exp(kde_eval))
    #     return numpy.exp(ll[0])

    from scipy.integrate import quad

    result = quad(kde_eval, -numpy.inf, numpy.inf)
    print(result)

if __name__ == '__main__':
    test_gaussian_kde_leaf()
