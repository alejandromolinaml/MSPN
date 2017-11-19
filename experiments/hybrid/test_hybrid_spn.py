'''
Created on Apr 20, 2017

@author: molina
'''


import numpy
from sklearn.metrics.classification import accuracy_score

from mlutils.datasets import getDiabetes, getAdult
from tfspn.SPN import SPN, Splitting
from tfspn.piecewise import estimate_domains


def test1():
    numpy.random.seed(42)
    data = numpy.random.poisson(5, 1000).reshape(1000, 1)

    for i in numpy.unique(data):
        print(i, numpy.sum(data == i))

    featureTypes = ["discrete"]
    featureTypes = ["categorical"]

    spn = SPN.LearnStructure(data, featureTypes=featureTypes, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.IndependenceTest(),
                             # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                             # domains, families=families, row_split_method=Splitting.KmeansRows(),
                             # col_split_method=Splitting.RDCTest(),
                             min_instances_slice=100)

    print(spn)
    print(numpy.unique(data))
    evdata = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    print(evdata)

    ll = (spn.root.eval(numpy.asarray(evdata).reshape(len(evdata), 1)))

    print("Probs", numpy.exp(ll))
    print("Sum LL", numpy.sum(ll))
    print(numpy.histogram(data, bins="auto", density=True))


def test2():
    numpy.random.seed(42)
    dsname, data, labels, classes, families = getDiabetes()

    labels = [l for l in labels]

    print(data.shape)

    print(data)
    featureTypes = ['discrete', 'continuous', 'continuous', 'continuous',
                    'continuous', 'continuous', 'continuous', 'continuous', 'continuous']
    featureTypes = ['continuous', 'categorical', 'continuous', 'continuous',
                    'continuous', 'continuous', 'continuous', 'continuous', 'continuous']
    # families[0] = 'bernoulli'

    # spn = SPN.LearnStructure(data, featureNames=labels, domains = domains,
    # families=families, row_split_method=Splitting.KmeansRows(),
    # col_split_method=Splitting.IndependenceTest(alpha=0.00001),
    spn = SPN.LearnStructure(data, featureTypes=featureTypes, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                             min_instances_slice=50, cluster_first=False)

    print(spn)
    # print(numpy.unique(data))

    ll = spn.root.eval(data)

    print("Sum LL", numpy.sum(ll))


def test3():
    numpy.random.seed(42)
    dsname, data, featureNames, featureTypes, doms = getAdult()

    doctorateVal = numpy.where(doms[2] == "Doctorate")[0][0]
    stategovVal = numpy.where(doms[1] == "State-gov")[0][0]

    print(featureNames)

    print(len(featureNames))

    print(data[0, :])
    print(data.shape)
    print(doctorateVal, stategovVal)

    pD = numpy.sum(data[:, 2] == doctorateVal) / data.shape[0]
    pSD = numpy.sum(numpy.logical_and(data[:, 2] == doctorateVal,
                                      data[:, 1] == stategovVal)) / data.shape[0]
    pS = numpy.sum(data[:, 1] == stategovVal) / data.shape[0]

    print("pD", pD)
    print("pSD", pSD)
    pS_D = pSD / pD
    print("pS_D", pS_D)

    # spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.IndependenceTest(alpha=0.01),
    # spn = SPN.LearnStructure(data, featureTypes=featureTypes,
    # featureNames=featureNames, row_split_method=Splitting.KmeansRows(),
    # col_split_method=Splitting.RDCTest(),
    spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                             min_instances_slice=3, cluster_first=True)

    spn.root.validate()

    print("SPN Learned")
    margSPN_SD = spn.root.marginalizeOut([0, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13])
    margSPN_SD.Prune()

    print(margSPN_SD)

    dataSD = numpy.zeros_like(data[0, :]).reshape(1, data.shape[1])
    dataSD[0, 1] = stategovVal
    dataSD[0, 2] = doctorateVal
    print(dataSD)

    spnSD = (numpy.exp(margSPN_SD.eval(dataSD)))

    margSPN_D = spn.root.marginalizeOut([0, 1, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13])
    margSPN_D.Prune()

    print(margSPN_D)

    dataD = numpy.zeros_like(data[0, :]).reshape(1, data.shape[1])
    dataD[0, 2] = doctorateVal
    print(dataD)

    spnD = (numpy.exp(margSPN_D.eval(dataD)))

    print("pD", pD)
    print("pS", pS)
    print("pSD", pSD)
    pS_D = pSD / pD
    print("pS_D", pS_D)

    print("spn pD", spnD)
    print("spn pSD", spnSD)
    spnS_D = spnSD / spnD
    print("spn pS_D", spnS_D)

    print("doctorateVal", doctorateVal)
    print("stategovVal", stategovVal)

    ll = spn.root.eval(data)

    # print("Probs", numpy.exp(ll))
    print("Sum LL", numpy.sum(ll))


def test4():
    numpy.random.seed(42)
    dsname, data, featureNames, featureTypes, doms = getAdult()

    data = data[:, [1, 2, 3, 4]]
    featureTypes = [featureTypes[1], featureTypes[2], featureTypes[3], featureTypes[4]]
    featureNames = [featureNames[1], featureNames[2], featureNames[3], featureNames[4]]
    doms = [doms[1], doms[2], doms[3], doms[4]]

    doctorateVal = numpy.where(doms[1] == "Doctorate")[0][0]
    stategovVal = numpy.where(doms[0] == "State-gov")[0][0]

    print(featureNames)

    print(data[0, :])
    print(doctorateVal, stategovVal)

    pD = numpy.sum(data[:, 1] == doctorateVal) / data.shape[0]
    pSD = numpy.sum(
        numpy.logical_and(data[:, 1] == doctorateVal, data[:, 0] == stategovVal)) / data.shape[0]
    pS = numpy.sum(data[:, 0] == stategovVal) / data.shape[0]

    print("pD", pD)
    print("pSD", pSD)
    pS_D = pSD / pD
    print("pS|D", pS_D)

    # spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.IndependenceTest(alpha=0.01),
    # spn = SPN.LearnStructure(data, featureTypes=featureTypes,
    # featureNames=featureNames, row_split_method=Splitting.KmeansRows(),
    # col_split_method=Splitting.RDCTest(),
    spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTestOHEpy(),
                             min_instances_slice=100, cluster_first=True)

    spn.root.validate()

    print("SPN Learned")
    margSPN_SD = spn.root.marginalizeOut([2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13])
    margSPN_SD.Prune()

    print(margSPN_SD)

    dataSD = numpy.zeros_like(data[0, :]).reshape(1, data.shape[1])
    dataSD[0, 0] = stategovVal
    dataSD[0, 1] = doctorateVal
    print(dataSD)

    spnSD = (numpy.exp(margSPN_SD.eval(dataSD)))

    margSPN_D = spn.root.marginalizeOut([0, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13])
    margSPN_D.Prune()

    print(margSPN_D)

    dataD = numpy.zeros_like(data[0, :]).reshape(1, data.shape[1])
    dataD[0, 1] = doctorateVal
    print(dataD)

    spnD = (numpy.exp(margSPN_D.eval(dataD)))

    print("pD", pD)
    print("pS", pS)
    print("pSD", pSD)
    pS_D = pSD / pD
    print("pS_D", pS_D)

    print("spn pD", spnD)
    print("spn pSD", spnSD)
    spnS_D = spnSD / spnD
    print("spn pS_D", spnS_D)

    print("doctorateVal", doctorateVal)
    print("stategovVal", stategovVal)

    ll = spn.root.eval(data)

    # print("Probs", numpy.exp(ll))
    print("Sum LL", numpy.sum(ll))


def test5():
    numpy.random.seed(42)

    data = numpy.zeros((5000, 2))

    idx = numpy.random.choice(data.shape[0], int(data.shape[0] / 2), replace=False)

    data[idx, 1] = 1

    idx0 = data[:, 1] == 0
    idx1 = data[:, 1] == 1

    data[idx0, 0] = numpy.random.normal(100, 30, numpy.sum(idx0))

    data[idx1, 0] = numpy.random.normal(200, 30, numpy.sum(idx1))

    print(data)

    featureNames = ["Gaussian", "Categorical"]
    featureTypes = ["continuous", "discrete"]

    # spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.IndependenceTest(alpha=0.01),
    # spn = SPN.LearnStructure(data, featureTypes=featureTypes,
    # featureNames=featureNames, row_split_method=Splitting.KmeansRows(),
    # col_split_method=Splitting.RDCTest(),
    spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTestOHEpy(),
                             min_instances_slice=500, cluster_first=True)

    spn.root.validate()

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import colorConverter
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)

    xs = np.arange(0, 300, 0.5)
    verts = []
    zs = [0, 1]

    maxys = 0
    for z in zs:
        testdata = numpy.zeros((len(xs), len(zs)))
        testdata[:, 0] = xs
        testdata[:, 1] = z

        ys = numpy.zeros_like(xs)

        ys[:] = numpy.exp(spn.root.eval(testdata))

        maxys = max(maxys, numpy.max(ys))

        ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))

    poly = PolyCollection(verts, facecolors=[cc('r'), cc('g')])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('X')
    ax.set_xlim3d(0, 300)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, 1)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, maxys)

    plt.show()

    ll = spn.root.eval(data)

    print("Sum LL", numpy.sum(ll))


def test6():
    numpy.random.seed(42)

    datablocks = []

    yd = [0, 1, 2, 3]
    xd = [0, 1]

    for x in xd:
        for y in yd:
            block = numpy.zeros((2000, 3))
            block[:, 1] = x
            block[:, 2] = y
            if (x == 1 and y == 0) or (x == 0 and y == 1) or (x == 1 and y == 2) or (x == 0 and y == 3):
                block[:, 0] = numpy.random.normal(200, 30, block.shape[0])
            else:
                block[:, 0] = numpy.random.normal(100, 30, block.shape[0])

            datablocks.append(block)

    data = numpy.vstack(datablocks)

    print(data.shape)

    featureNames = ["Gaussian", "Categorical", "Discrete"]
    featureTypes = ["continuous", "categorical", "discrete"]

    # spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.IndependenceTest(alpha=0.01),
    # spn = SPN.LearnStructure(data, featureTypes=featureTypes,
    # featureNames=featureNames, row_split_method=Splitting.KmeansRows(),
    # col_split_method=Splitting.RDCTest(),
    spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTestOHEpy(),
                             min_instances_slice=50, cluster_first=True)

    spn.root.validate()

    from matplotlib.collections import PolyCollection
    from matplotlib.colors import colorConverter
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(len(xd), len(yd))

    fig = plt.figure(figsize=(8, 8))

    xall = numpy.arange(0, 300, 0.5)
    i = 0
    for x in xd:
        for y in yd:
            testdata = numpy.zeros((len(xall), 3))
            testdata[:, 0] = xall
            testdata[:, 1] = x
            testdata[:, 2] = y

            pbs = numpy.zeros_like(xall)

            pbs[:] = numpy.exp(spn.root.eval(testdata))

            ax = plt.Subplot(fig, gs[i])
            i += 1

            ax.set_title('%s %s' % (x, y))
            ax.plot(xall, pbs, 'r--')

            fig.add_subplot(ax)

    plt.show()

    ll = spn.root.eval(data)

    print("Sum LL", numpy.sum(ll))


def test7():
    numpy.random.seed(42)

    D = numpy.loadtxt("bank.csv", delimiter=";", skiprows=0, dtype="S")
    D = numpy.char.strip(D)

    featureNames = [str(f) for f in D[0, :]]
    D = D[1:, :]
    featureTypes = ["discrete", "categorical", "categorical", "categorical", "continuous", "continuous",
                    "categorical", "categorical", "categorical", "discrete", "categorical", "discrete",
                    "categorical", "continuous", "discrete", "categorical", "categorical", ]
    print(len(featureTypes))
    print(len(featureNames))

    def isinteger(x):
        return numpy.all(numpy.equal(numpy.mod(x, 1), 0))

    cols = []
    types = []
    domains = []

    index = [0, 5]

    D = D[:, index]

    for col in range(D.shape[1]):
        b, c = numpy.unique(D[:, col], return_inverse=True)

        try:
            # could convert to float
            if isinteger(b.astype(float)):
                # was integer
                cols.append(D[:, col].astype(int))
                types.append("discrete")
                domains.append(b.astype(int))
                continue

            # was float
            cols.append(D[:, col].astype(float))
            types.append("continuous")
            domains.append(b.astype(float))

            continue
        except:
            # was discrete
            cols.append(c)
            types.append("categorical")
            domains.append(b.astype(str))

    data = numpy.column_stack(cols)
    print(featureNames)

    print(domains)
    featureNames = [featureNames[i] for i in index]
    print(featureNames)
    print(types)

    data[:, 1] = numpy.sign(data[:, 1]) * numpy.log(numpy.abs(data[:, 1]) + 1)

    # spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.IndependenceTest(alpha=0.01),
    # spn = SPN.LearnStructure(data, featureTypes=featureTypes,
    # featureNames=featureNames, row_split_method=Splitting.KmeansRows(),
    # col_split_method=Splitting.RDCTest(),
    spn = SPN.LearnStructure(data, featureTypes=types, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTest(threshold=0.000001),
                             min_instances_slice=1000, cluster_first=False)
    # RDCTestOHEpy

    spn.root.validate()

    print(spn)

    spn.save_pdf_graph("bank.pdf")

    ll = spn.root.eval(data)

    from matplotlib.collections import PolyCollection
    from matplotlib.colors import colorConverter
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    for i in [0, 1]:

        x = numpy.sort(data[:, i]).reshape(data.shape[0], 1)

        fig = plt.figure(figsize=(8, 8))
        x1 = numpy.zeros_like(data)
        x1[:, i] = x[:, 0]

        color_idx = numpy.linspace(0, 1, len(spn.root.children))

        for cidx, c in enumerate(spn.root.children):

            y = numpy.exp(c.children[i].eval(x1))

            plt.plot(x, y, '--', color=plt.cm.cool(color_idx[cidx]))

    plt.show()

    # print("Probs", numpy.exp(ll))
    print("Sum LL", numpy.sum(ll))


def test8():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    data, target = mnist.train.images, mnist.train.labels

    featureTypes = ["continuous"] * data.shape[1] + ["categorical"]

    featureNames = ["pixel"] * data.shape[1] + ["label"]

    data = numpy.hstack((data, target.reshape(data.shape[0], 1)))
    print(featureTypes)
    print(data.shape)

    spn = SPN.LearnStructure(data, featureTypes=featureTypes, featureNames=featureNames, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTest(threshold=0.4),
                             min_instances_slice=500, cluster_first=True)
    # RDCTestOHEpy

    print("learned")

    spn.root.validate()

    data, target = mnist.test.images, mnist.test.labels

    data = numpy.hstack((data, target.reshape(data.shape[0], 1)))

    classes = numpy.unique(target)
    results = numpy.zeros((data.shape[0], len(classes)))

    print("testing")
    # print(spn)
    for c in classes:
        data[:, -1] = c
        results[:, c] = spn.root.eval(data)

    print("done")

    predictions = numpy.argmax(results, axis=1)

    print('MAP accuracy : ', accuracy_score(target, predictions))

    # print("Probs", numpy.exp(ll))
    #print("Sum LL", numpy.sum(ll))

if __name__ == '__main__':
    test8()
