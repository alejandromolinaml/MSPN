from matplotlib import pyplot


def test_piecewise_marginals():

    from mlutils.datasets import loadMLC
    from tfspn.piecewise import piecewise_linear_approximation, estimate_bins, estimate_domains
    from tfspn.tfspn import SumNode, ProductNode, PoissonNode, GaussianNode, BernoulliNode, \
        PiecewiseLinearPDFNode

    #
    # loading australian
    (train, test, valid), fnames, ftypes, domains = loadMLC('australian')

    print(train.shape)
    print(test.shape)
    print(valid.shape)

    for fn, ft, fd in zip(fnames, ftypes, domains):
        print(fn, ft, fd[:2], fd[-2:])

    #
    # some continuous features
    # A2, A3, A7, A10
    # c_feature_ids = [1, 2, 6, 9]
    c_feature_ids = [1]

    #
    #
    n_bins = 100
    for i in c_feature_ids:

        train_data = train[:, i]
        valid_data = valid[:, i]
        test_data = test[:, i]

        # pyplot.hist(train_data, bins=n_bins, alpha=0.4, label='train', normed=True)
        # pyplot.hist(valid_data, bins=n_bins, alpha=0.4, label='valid', normed=True)
        # pyplot.hist(test_data, bins=n_bins, alpha=0.4, label='test', normed=True)
        # pyplot.legend(loc='upper right')
        # pyplot.show()

        #
        # creating a piecewise node
        print('looking at feature', i, ftypes[i], domains[i])
        # bins = estimate_bins(train_data, ftypes[i], domains[i])
        print(train_data.min(), train_data.max())
        # print('computed bins', bins)
        smoothing = 1
        print('domains', domains[i])
        bins = estimate_bins(train_data, ftypes[i], [domains[i]])
        print('bins from domains', bins)
        x_range, y_range = piecewise_linear_approximation(train_data,
                                                          bins=bins,
                                                          family=ftypes[i],
                                                          alpha=smoothing,
                                                          # isotonic=True,
                                                          # n_bootstraps=nb,
                                                          # average_bootstraps=False
                                                          )
        # print("PiecewiseLinearPDFNode")
        node = PiecewiseLinearPDFNode("PiecewiseLinearNode_{}".format(i),
                                      i,
                                      fnames[i],
                                      domains[i],
                                      x_range, y_range)
        print(node)

        #
        # compute likelihoods
        train_lls = node.eval(train)
        valid_lls = node.eval(valid)
        test_lls = node.eval(test)

        print('TRAIN LL:', train_lls.mean())
        print('VALID LL:', valid_lls.mean())
        print('TEST LL:', test_lls.mean())

        v_ids = valid_data > 76.75
        print(sum(v_ids), valid_lls[v_ids])

        t_ids = test_data > 76.75
        print(sum(t_ids), test_lls[t_ids])
        print(test_lls)
        print(test_lls[~t_ids].mean())

if __name__ == '__main__':
    test_piecewise_marginals()
