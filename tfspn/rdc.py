"""
Created on Apr 10, 2017

@author: antonio vergari
"""


import itertools

from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
import numpy
import scipy.stats
from sklearn.cross_decomposition import CCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from multiprocessing import Pool
import concurrent.futures


RAND_STATE = 1337

GLOBAL_RDC_FEATURES = []


def ecdf(X):
    """
    Empirical cumulative distribution function
    for data X (one dimensional, if not it is linearized first)
    """
    return scipy.stats.rankdata(X, method='max') / len(X)


def rdc(X, Y, k=None, s=1. / 6., f=numpy.sin, rand_gen=None, rnorm_X=None, rnorm_Y=None):

    if X.ndim == 1:
        X = X[:, numpy.newaxis]
    if Y.ndim == 1:
        Y = Y[:, numpy.newaxis]

    #
    # heuristic assumption
    if k is None:
        k = max(X.shape[1], Y.shape[1]) + 1
        # print(k)

    n_instances = X.shape[0]
    assert Y.shape[0] == n_instances, (Y.shape[0], n_instances)

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_STATE)

    #
    # empirical copula transformation
    ones_column = numpy.ones((n_instances, 1))
    X_c = numpy.concatenate((numpy.apply_along_axis(ecdf, 0, X),
                             ones_column), axis=1)
    Y_c = numpy.concatenate((numpy.apply_along_axis(ecdf, 0, Y),
                             ones_column), axis=1)

    #
    # linear projection through a random gaussian
    if rnorm_X is None:
        rnorm_X = rand_gen.normal(size=(X_c.shape[1], k))
    if rnorm_Y is None:
        rnorm_Y = rand_gen.normal(size=(Y_c.shape[1], k))
    X_proj = s / X_c.shape[1] * numpy.dot(X_c, rnorm_X)
    Y_proj = s / Y_c.shape[1] * numpy.dot(Y_c, rnorm_Y)

    #
    # non-linear projection
    # print(f(X_proj), f(X_proj).shape, X_proj.shape)
    X_proj = numpy.concatenate((f(X_proj), ones_column), axis=1)
    Y_proj = numpy.concatenate((f(Y_proj), ones_column), axis=1)

    #
    # canonical correlation analysis
    cca = CCA(n_components=1)
    X_cca, Y_cca = cca.fit_transform(X_proj, Y_proj)

    rdc = numpy.corrcoef(X_cca.T, Y_cca.T)

    # print(rdc)
    return rdc[0, 1]


def make_matrix(data):
    """
    Ensures data to be 2-dimensional
    """
    if data.ndim == 1:
        data = data[:, numpy.newaxis]
    else:
        assert data.ndim == 2, "Data must be 2 dimensional {}".format(data.shape)

    return data


def empirical_copula_transformation(data):
    ones_column = numpy.ones((data.shape[0], 1))
    data = numpy.concatenate((numpy.apply_along_axis(ecdf, 0, data),
                              ones_column), axis=1)
    return data


def rdc_transformer(data_slice,
                    k=None,
                    s=1. / 6.,
                    non_linearity=numpy.sin,
                    return_matrix=False,
                    ohe=True,
                    rand_gen=None):

    #print('rdc transformer', k, s, non_linearity)
    """
    Given a data_slice,
    return a transformation of the features data in it according to the rdc
    pipeline:
    1 - empirical copula transformation
    2 - random projection into a k-dimensional gaussian space
    3 - pointwise  non-linear transform
    """

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_STATE)

    #
    # precomputing transformations to reduce time complexity
    #

    if ohe:
        features = [data_slice.getOHEFeatureData(f) for f in data_slice.cols]
    else:
        features = [data_slice.getFeatureData(f) for f in data_slice.cols]

    #
    # NOTE: here we are setting a global k for ALL features
    # to be able to precompute gaussians
    if k is None:
        feature_shapes = [f.shape[1] if len(f.shape) > 1 else 1 for f in features]
        k = max(feature_shapes) + 1

    #
    # forcing two columness
    features = [make_matrix(f) for f in features]

    #
    # transform through the empirical copula
    features = [empirical_copula_transformation(f) for f in features]

    #
    # random projection through a gaussian
    random_gaussians = [rand_gen.normal(size=(f.shape[1], k))
                        for f in features]

    rand_proj_features = [s / f.shape[1] * numpy.dot(f, N)
                          for f, N in zip(features,
                                          random_gaussians)]

    nl_rand_proj_features = [non_linearity(f)
                             for f in rand_proj_features]

    #
    # apply non-linearity
    if return_matrix:
        return numpy.concatenate(nl_rand_proj_features, axis=1)

    else:
        print([f.shape for f in nl_rand_proj_features])
        return [numpy.concatenate((f, numpy.ones((f.shape[0], 1))), axis=1)
                for f in nl_rand_proj_features]


# def rdc_cca(i, j):
def rdc_cca(indexes):
    i, j, cca = indexes
    cca = CCA(n_components=1)
    X_cca, Y_cca = cca.fit_transform(GLOBAL_RDC_FEATURES[i],
                                     GLOBAL_RDC_FEATURES[j])
    # rdc = 1
    rdc = numpy.corrcoef(X_cca.T, Y_cca.T)[0, 1]
    print('ij', i, j)
    return rdc


def f(i, j):
    return i + j


def rdc_test(data_slice,
             k=None,
             s=1. / 6.,
             non_linearity=numpy.sin,
             n_jobs=7,
             rand_gen=None):

    n_features = len(data_slice.cols)

    if n_jobs is None:
        n_jobs = n_features * n_features

    rdc_features = rdc_transformer(data_slice,
                                   k=k,
                                   s=s,
                                   non_linearity=non_linearity,
                                   return_matrix=False,
                                   rand_gen=rand_gen)
    #
    # build adjacency matrix
    rdc_adjacency_matrix = numpy.zeros((n_features, n_features))

    GLOBAL_RDC_FEATURES.clear()
    GLOBAL_RDC_FEATURES.extend(rdc_features)

    pairwise_comparisons = itertools.combinations(numpy.arange(n_features), 2)
    rdc_vals = None
    # with Pool(n_jobs) as p:
    # p = Pool(n_jobs)

    cca = CCA(n_components=1)
    from joblib import Parallel, delayed
    rdc_vals = Parallel(n_jobs=n_jobs)(delayed(rdc_cca)((i, j, cca))
                                       for i, j in pairwise_comparisons)

    # with concurrent.futures.ProcessPoolExecutor(n_jobs) as p:
    #     rdc_vals = p.map(rdc_cca, [(i, j) for i, j in pairwise_comparisons])
    # rdc_vals = []
    # for i, j in pairwise_comparisons:
    #     rdc_vals.append(rdc_cca((i, j)))

    pairwise_comparisons = itertools.combinations(numpy.arange(n_features), 2)
    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        print(rdc)
        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc

    #
    # setting diagonal to 1
    rdc_adjacency_matrix[numpy.diag_indices_from(rdc_adjacency_matrix)] = 1
    print(rdc_adjacency_matrix)

    return rdc_adjacency_matrix


def getIndependentRDCGroups_py(data_slice,
                               threshold,
                               k=None,
                               s=1. / 6.,
                               non_linearity=numpy.sin,
                               n_jobs=1,
                               rand_gen=None):

    rdc_adjacency_matrix = rdc_test(data_slice,
                                    k=k,
                                    s=s,
                                    non_linearity=non_linearity,
                                    n_jobs=n_jobs,
                                    rand_gen=rand_gen)

    n_features = len(data_slice.cols)

    #
    # thresholding
    rdc_adjacency_matrix[rdc_adjacency_matrix < threshold] = 0
    #print("thresholding", rdc_adjacency_matrix)

    #
    # getting connected components
    result = numpy.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix))):
        result[list(c)] = i + 1

    return result


def standardize_and_discretize(data, float_dtype=numpy.float64):

    #
    # ensure continuous representation
    data = numpy.array(data).astype(float_dtype)

    #
    # standardizing
    data = (data - numpy.mean(data)) / numpy.std(data)

    #
    # discretizing with "Freedman Diaconis Estimator"
    data = numpy.digitize(data, numpy.histogram(data, bins='fd')[1])

    return data


def guyon_independence_test(x, y, x_type, y_type, float_dtype=numpy.float64):

    #
    # preprocess continuous features
    if x_type == 'continuous':
        x = standardize_and_discretize(x, float_dtype=float_dtype)

    if y_type == 'continuous':
        y = standardize_and_discretize(y, float_dtype=float_dtype)

    #
    #
    dep_coef = adjusted_mutual_info_score(x, y)

    return dep_coef


def pairwise_gdt(data_slice,
                 # n_jobs=1,
                 # rand_gen=None,
                 float_dtype=numpy.float64):

    n_features = len(data_slice.cols)

    # build adjacency matrix
    gdt_adjacency_matrix = numpy.zeros((n_features, n_features))

    pairwise_comparisons = itertools.combinations(numpy.arange(n_features), 2)

    for i, j in pairwise_comparisons:

        x = data_slice.getFeatureData(data_slice.cols[i])
        x = x.reshape(x.shape[0])
        y = data_slice.getFeatureData(data_slice.cols[j])
        y = y.reshape(y.shape[0])

        x_type = data_slice.families[i]
        y_type = data_slice.families[j]

        gdt = guyon_independence_test(x, y, x_type, y_type, float_dtype=float_dtype)

        gdt_adjacency_matrix[i, j] = gdt
        gdt_adjacency_matrix[j, i] = gdt

    #
    # setting diagonal to 1
    gdt_adjacency_matrix[numpy.diag_indices_from(gdt_adjacency_matrix)] = 1
    #print(gdt_adjacency_matrix)

    return gdt_adjacency_matrix


def getIndependentGDTGroups_py(data_slice,
                               threshold,
                               # n_jobs=1,
                               rand_gen=None):

    gdt_adjacency_matrix = pairwise_gdt(data_slice,
                                        )

    n_features = len(data_slice.cols)

    #
    # thresholding
    gdt_adjacency_matrix[gdt_adjacency_matrix < threshold] = 0
    #print("thresholding", gdt_adjacency_matrix)

    #
    # getting connected components
    result = numpy.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(gdt_adjacency_matrix))):
        result[list(c)] = i + 1

    return result
