import sys

import numpy
from numpy.testing import assert_array_equal

from tfspn.SPN import DataSlice
from tfspn.SPN import Splitting
from tfspn.piecewise import estimate_domains


def test_random_balanced_binary_split():

    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    rand_gen = numpy.random.RandomState(1337)

    n_instances = 100
    n_features = 6
    feature_types = ["continuous", "categorical",
                     "discrete", "categorical", "continuous", "discrete"]
    data = numpy.array([rand_gen.randn(n_instances),
                        rand_gen.choice(4, size=n_instances),
                        rand_gen.choice(numpy.arange(-10, 10), size=n_instances),
                        rand_gen.choice(6, size=n_instances),
                        rand_gen.randn(n_instances),
                        rand_gen.choice(numpy.arange(-20, 20), size=n_instances)]).T

    print('data', data)

    domains = estimate_domains(data, feature_types)
    print('domains', domains)

    families = ['piecewise'] * len(domains)
    names = ['f{}'.format(f) for f in range(len(domains))]

    #
    # subset of features
    rows = rand_gen.choice(n_instances, size=10)
    cols = numpy.array([3, 4, 5])
    data_slice = DataSlice(data, families=families, domains=domains, featureNames=names,
                           featureTypes=feature_types,
                           rows=rows,
                           cols=cols)
    print(data_slice)
    print('filtered data', data_slice.getData())

    config = {'rand_gen': rand_gen}
    clustering_func = Splitting.GetFunction('Random Balanced Binary Split', config)

    data_slice_clusters, n_clusters = clustering_func(data_slice)
    print(data_slice_clusters)

    assert n_clusters == 2
    assert len(data_slice_clusters) == 2
    abs(len(data_slice_clusters[0].rows) - len(data_slice_clusters[1].rows)) <= 1
    assert_array_equal(numpy.sort(numpy.union1d(data_slice_clusters[0].rows, data_slice_clusters[1].rows)),
                       numpy.sort(rows))
    assert_array_equal(numpy.sort(cols), numpy.sort(data_slice_clusters[0].cols))
    assert_array_equal(numpy.sort(cols), numpy.sort(data_slice_clusters[1].cols))


def test_random_binary_conditioning_split():

    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    rand_gen = numpy.random.RandomState(1337)

    n_instances = 100
    n_features = 6
    feature_types = ["continuous", "categorical",
                     "discrete", "categorical", "continuous", "discrete"]
    data = numpy.array([rand_gen.randn(n_instances),
                        rand_gen.choice(4, size=n_instances),
                        rand_gen.choice(numpy.arange(-10, 10), size=n_instances),
                        rand_gen.choice(6, size=n_instances),
                        rand_gen.randn(n_instances),
                        rand_gen.choice(numpy.arange(-20, 20), size=n_instances)]).T

    print('data', data)

    domains = estimate_domains(data, feature_types)
    print('domains', domains)

    families = ['piecewise'] * len(domains)
    names = ['f{}'.format(f) for f in range(len(domains))]

    #
    # subset of features
    rows = rand_gen.choice(n_instances, size=10)
    cols = numpy.array([3, 4, 5])
    data_slice = DataSlice(data, families=families, domains=domains, featureNames=names,
                           featureTypes=feature_types,
                           rows=rows,
                           cols=cols)
    print(data_slice)
    print('filtered data', data_slice.getData())

    config = {'rand_gen': rand_gen}
    clustering_func = Splitting.GetFunction('Random Binary Conditioning Split', config)

    data_slice_clusters, n_clusters = clustering_func(data_slice)
    print(data_slice_clusters)

    assert n_clusters == 2
    assert len(data_slice_clusters) == 2
    abs(len(data_slice_clusters[0].rows) - len(data_slice_clusters[1].rows)) <= 1
    assert_array_equal(numpy.sort(numpy.union1d(data_slice_clusters[0].rows, data_slice_clusters[1].rows)),
                       numpy.sort(rows))
    assert_array_equal(numpy.sort(cols), numpy.sort(data_slice_clusters[0].cols))
    assert_array_equal(numpy.sort(cols), numpy.sort(data_slice_clusters[1].cols))


if __name__ == '__main__':
    test_random_balanced_binary_split()
    # test_random_binary_conditioning_split()
