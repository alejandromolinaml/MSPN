from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import numpy

from tfspn.rdc import rdc
from tfspn.rdc import rdc_transformer
from tfspn.rdc import rdc_test
from tfspn.rdc import getIndependentRDCGroups_py


RRDC_rank = """
function(x,y,Nx,Ny,k=20,s=1/6,f=sin) {
  x <- cbind(apply(as.matrix(x),2,function(u)rank(u)/length(u)),1)
  y <- cbind(apply(as.matrix(y),2,function(u)rank(u)/length(u)),1)
  x <- s/ncol(x)*x%*%matrix(Nx,ncol(x))
  y <- s/ncol(y)*y%*%matrix(Ny,ncol(y))
  cancor(cbind(f(x),1),cbind(f(y),1))$cor[1]
}
"""

RRDC_ecdf = """
function(x,y,Nx,Ny,k=20,s=1/6,f=sin) {
  x <- cbind(apply(as.matrix(x),2,function(u)ecdf(u)(u)),1)
  y <- cbind(apply(as.matrix(y),2,function(u)ecdf(u)(u)),1)
  x <- s/ncol(x)*x%*%matrix(Nx,ncol(x))
  y <- s/ncol(y)*y%*%matrix(Ny,ncol(y))
  cancor(cbind(f(x),1),cbind(f(y),1))$cor[1]
}
"""


def test_pyrdc_vs_Rrdc_normal_data():

    numpy2ri.activate()

    #
    # generate random data from two gaussians
    rand_gen = numpy.random.RandomState(1337)

    n_instances = 5000
    X_cols = 1
    X_size = (n_instances, X_cols)
    X_loc = 10
    X_var = 1.0
    X_rand = rand_gen.normal(size=X_size, loc=X_loc, scale=X_var)

    Y_cols = 1
    Y_size = (n_instances, Y_cols)
    Y_loc = -25
    Y_var = 0.25
    Y_rand = rand_gen.normal(size=Y_size, loc=Y_loc, scale=Y_var)

    print('X: {}, Y: {}'.format(X_rand.shape, Y_rand.shape))

    k_values = [1, 2, 10, 20, Y_cols, X_cols, X_cols + Y_cols, (X_cols + Y_cols) * 2]

    for k in k_values:
        print('\nConsidering k: {}'.format(k))
        #
        # generate random normals through R
        rnorm = robjects.r["rnorm"]
        rnorm_X = rnorm((X_cols + 1) * k)
        rnorm_X_m = numpy.array(rnorm_X).reshape(X_cols + 1, k)
        rnorm_Y = rnorm((Y_cols + 1) * k)
        rnorm_Y_m = numpy.array(rnorm_Y).reshape(Y_cols + 1, k)
        print("\trnorm X: {}, rnorm Y: {}".format(rnorm_X_m.shape,
                                                  rnorm_Y_m.shape))

        #
        # loading the R rdc test test from
        Rrdc_rank = robjects.r(RRDC_rank)

        Rrdc_rank_value = Rrdc_rank(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (rank) value: {}".format(Rrdc_rank_value))

        #
        # loading the R rdc test test from
        Rrdc_ecdf = robjects.r(RRDC_ecdf)

        Rrdc_ecdf_value = Rrdc_ecdf(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (ecdf) value: {}".format(Rrdc_ecdf_value))

        pyrdc_value = rdc(X_rand, Y_rand, rnorm_X=rnorm_X_m, rnorm_Y=rnorm_Y_m)
        print("\tPython value: {}".format(pyrdc_value))


def test_pyrdc_vs_Rrdc_normal_dependent_data():

    numpy2ri.activate()

    #
    # generate random data from two gaussians
    rand_gen = numpy.random.RandomState(1337)

    n_instances = 5000
    X_cols = 100
    X_size = (n_instances, X_cols)
    X_loc = 10
    X_var = 1.0
    X_rand = rand_gen.normal(size=X_size, loc=X_loc, scale=X_var)

    Y_cols = 100
    Y_size = (n_instances, Y_cols)
    Y_loc = -25
    Y_var = 0.25
    alpha = 3.0
    beta = 115
    Y_rand = alpha * X_rand + beta  # rand_gen.normal(size=Y_size, loc=Y_loc, scale=Y_var)

    print('X: {}, Y: {}'.format(X_rand.shape, Y_rand.shape))

    k_values = [1, 2, 10, 20, Y_cols, X_cols, X_cols + Y_cols, (X_cols + Y_cols) * 2]

    for k in k_values:
        print('\nConsidering k: {}'.format(k))
        #
        # generate random normals through R
        rnorm = robjects.r["rnorm"]
        rnorm_X = rnorm((X_cols + 1) * k)
        rnorm_X_m = numpy.array(rnorm_X).reshape(X_cols + 1, k)
        rnorm_Y = rnorm((Y_cols + 1) * k)
        rnorm_Y_m = numpy.array(rnorm_Y).reshape(Y_cols + 1, k)
        print("\trnorm X: {}, rnorm Y: {}".format(rnorm_X_m.shape,
                                                  rnorm_Y_m.shape))

        #
        # loading the R rdc test test from
        Rrdc_rank = robjects.r(RRDC_rank)

        Rrdc_rank_value = Rrdc_rank(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (rank) value: {}".format(Rrdc_rank_value))

        #
        # loading the R rdc test test from
        Rrdc_ecdf = robjects.r(RRDC_ecdf)

        Rrdc_ecdf_value = Rrdc_ecdf(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (ecdf) value: {}".format(Rrdc_ecdf_value))

        pyrdc_value = rdc(X_rand, Y_rand, rnorm_X=rnorm_X_m, rnorm_Y=rnorm_Y_m)
        print("\tPython value: {}".format(pyrdc_value))


def test_pyrdc_vs_Rrdc_bernoulli_data():

    numpy2ri.activate()

    #
    # generate random data from two gaussians
    rand_gen = numpy.random.RandomState(1337)

    n_instances = 5000
    X_cols = 100
    X_size = (n_instances, X_cols)
    X_theta = .68
    X_rand = rand_gen.binomial(n=1, p=X_theta, size=X_size)

    Y_cols = 50
    Y_size = (n_instances, Y_cols)
    Y_theta = .22
    Y_rand = rand_gen.binomial(n=1, p=Y_theta, size=Y_size)

    print('X: {}, Y: {}'.format(X_rand.shape, Y_rand.shape))

    k_values = [1, 2, 10, 20, Y_cols, X_cols, X_cols + Y_cols, (X_cols + Y_cols) * 2]

    for k in k_values:
        print('\nConsidering k: {}'.format(k))
        #
        # generate random normals through R
        rnorm = robjects.r["rnorm"]
        rnorm_X = rnorm((X_cols + 1) * k)
        rnorm_X_m = numpy.array(rnorm_X).reshape(X_cols + 1, k)
        rnorm_Y = rnorm((Y_cols + 1) * k)
        rnorm_Y_m = numpy.array(rnorm_Y).reshape(Y_cols + 1, k)
        print("\trnorm X: {}, rnorm Y: {}".format(rnorm_X_m.shape,
                                                  rnorm_Y_m.shape))

        #
        # loading the R rdc test test from
        Rrdc_rank = robjects.r(RRDC_rank)

        Rrdc_rank_value = Rrdc_rank(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (rank) value: {}".format(Rrdc_rank_value))

        #
        # loading the R rdc test test from
        Rrdc_ecdf = robjects.r(RRDC_ecdf)

        Rrdc_ecdf_value = Rrdc_ecdf(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (ecdf) value: {}".format(Rrdc_ecdf_value))

        pyrdc_value = rdc(X_rand, Y_rand, rnorm_X=rnorm_X_m, rnorm_Y=rnorm_Y_m)
        print("\tPython value: {}".format(pyrdc_value))


def test_pyrdc_vs_Rrdc_bernoulli_data_monodim():

    numpy2ri.activate()

    #
    # generate random data from two gaussians
    rand_gen = numpy.random.RandomState(1337)

    n_instances = 20
    X_cols = 1
    X_size = (n_instances, X_cols)
    X_theta = .68
    X_rand = rand_gen.binomial(n=1, p=X_theta, size=X_size)
    print(','.join(str(c[0]) for c in X_rand))

    Y_cols = 1
    Y_size = (n_instances, Y_cols)
    Y_theta = .22
    Y_rand = rand_gen.binomial(n=1, p=Y_theta, size=Y_size)
    print(','.join(str(c[0]) for c in Y_rand))

    print('X: {}, Y: {}'.format(X_rand.shape, Y_rand.shape))

    k_values = [1, 2, 10, 20, 100, 200]

    for k in k_values:
        print('\nConsidering k: {}'.format(k))
        #
        # generate random normals through R
        rnorm = robjects.r["rnorm"]
        rnorm_X = rnorm((X_cols + 1) * k)
        rnorm_X_m = numpy.array(rnorm_X).reshape(X_cols + 1, k)
        rnorm_Y = rnorm((Y_cols + 1) * k)
        rnorm_Y_m = numpy.array(rnorm_Y).reshape(Y_cols + 1, k)
        print("\trnorm X: {}, rnorm Y: {}".format(rnorm_X_m.shape,
                                                  rnorm_Y_m.shape))

        #
        # loading the R rdc test test from
        Rrdc_rank = robjects.r(RRDC_rank)

        Rrdc_rank_value = Rrdc_rank(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (rank) value: {}".format(Rrdc_rank_value))

        #
        # loading the R rdc test test from
        Rrdc_ecdf = robjects.r(RRDC_ecdf)

        Rrdc_ecdf_value = Rrdc_ecdf(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (ecdf) value: {}".format(Rrdc_ecdf_value))

        pyrdc_value = rdc(X_rand, Y_rand, rnorm_X=rnorm_X_m, rnorm_Y=rnorm_Y_m)
        print("\tPython value: {}".format(pyrdc_value))


def test_pyrdc_vs_Rrdc_one_hot_enc():

    numpy2ri.activate()

    #
    # generate random data from two gaussians
    rand_gen = numpy.random.RandomState(1337)

    n_instances = 20
    X_cols = 10
    X_size = (n_instances, X_cols)
    X_rand = numpy.zeros(X_size)
    for i in range(n_instances):
        k = rand_gen.choice(X_cols)
        X_rand[i, k] = 1

    print(X_rand)

    Y_cols = 10
    Y_size = (n_instances, Y_cols)
    Y_rand = numpy.zeros(Y_size)
    for i in range(n_instances):
        k = rand_gen.choice(Y_cols)
        Y_rand[i, k] = 1
    print(Y_rand)

    print('X: {}, Y: {}'.format(X_rand.shape, Y_rand.shape))

    k_values = [1, 2, 10, 20, 100, 200]

    for k in k_values:
        print('\nConsidering k: {}'.format(k))
        #
        # generate random normals through R
        rnorm = robjects.r["rnorm"]
        rnorm_X = rnorm((X_cols + 1) * k)
        rnorm_X_m = numpy.array(rnorm_X).reshape(X_cols + 1, k)
        rnorm_Y = rnorm((Y_cols + 1) * k)
        rnorm_Y_m = numpy.array(rnorm_Y).reshape(Y_cols + 1, k)
        print("\trnorm X: {}, rnorm Y: {}".format(rnorm_X_m.shape,
                                                  rnorm_Y_m.shape))

        #
        # loading the R rdc test test from
        Rrdc_rank = robjects.r(RRDC_rank)

        Rrdc_rank_value = Rrdc_rank(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (rank) value: {}".format(Rrdc_rank_value))

        #
        # loading the R rdc test test from
        Rrdc_ecdf = robjects.r(RRDC_ecdf)

        Rrdc_ecdf_value = Rrdc_ecdf(X_rand, Y_rand, rnorm_X, rnorm_Y, k=k)
        print("\tR (ecdf) value: {}".format(Rrdc_ecdf_value))

        pyrdc_value = rdc(X_rand, Y_rand, rnorm_X=rnorm_X_m, rnorm_Y=rnorm_Y_m)
        print("\tPython value: {}".format(pyrdc_value))


def test_getIndependentRDCGroups_py_normal_data():

    #
    # generate random data from two gaussians
    rand_gen = numpy.random.RandomState(1337)

    n_instances = 5000
    X_cols = 10
    X_size = (n_instances, X_cols)
    X_loc = 10
    X_var = 1.0
    X_rand = rand_gen.normal(size=X_size, loc=X_loc, scale=X_var)
    families = ['gaussian' for i in range(X_cols)]
    domains = [None for i in range(X_cols)]
    feature_names = ['g{}'.format(i) for i in range(X_cols)]
    feature_types = ['continuous' for i in range(X_cols)]

    print('X: {}'.format(X_rand.shape))

    k_values = [1, 2, 3]

    #
    # creating data slice
    from tfspn.SPN import DataSlice

    data_slice = DataSlice(data=X_rand, families=families,
                           domains=domains,
                           featureNames=feature_names,
                           featureTypes=feature_types,
                           rows=numpy.arange(n_instances),
                           cols=numpy.arange(X_cols))

    for k in k_values:
        print('\nConsidering k: {}'.format(k))

        # rdc_matrix = rdc_test(data_slice,
        #                       k=k,
        #                       s=1. / 6.,
        #                       rand_gen=rand_gen)

        # print('py rdc_matrix', rdc_matrix)

        res = getIndependentRDCGroups_py(data_slice,
                                         threshold=0.05,
                                         k=k,
                                         s=1. / 6.,
                                         rand_gen=rand_gen)

    print(res)

test_getIndependentRDCGroups_py_normal_data()
