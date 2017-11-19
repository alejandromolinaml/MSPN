'''
Created on May 22, 2017

@author: molina
'''




import dirichlet
from dirichlet.simplex import contour, contourf, cartesian
from joblib.memory import Memory
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.path import Path
import numpy
import pandas
import scipy.stats
from sklearn.decomposition.online_lda import LatentDirichletAllocation
from sklearn.model_selection._split import train_test_split

from experiments.archetypes.archetypes import getArchetypes, getDirichlet
from experiments.archetypes.coordinates import polycorners, cart2bary
from experiments.archetypes.dirplot import Dirichlet2plot, draw_pdf_contours, \
    draw_pdf_contours_func, draw_pdf_contours_func2
from experiments.synth.test import rdccancor, rdcdcor
import matplotlib.pyplot as plt
from mlutils.benchmarks import Stats, Chrono
from mlutils.datasets import getNips, estimate_continuous_domain, getTraffic, \
    getHydrochem, getAirQualityUCITimeless
from mlutils.test import kfolded
from tfspn.SPN import SPN, Splitting
from tfspn.histogram import getHistogramVals
from tfspn.tfspn import ProductNode, SumNode
from multiprocessing import Pool

numpy.set_printoptions(precision=4, suppress=True)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

memory = Memory(cachedir="/tmp/archetypes", verbose=0, compress=9)

def getAirPollution(dimensions):
    dsname, data, features = getAirQualityUCITimeless()
    
    idxmissing = data == -200
    
    data = data[:, numpy.sum(idxmissing,0) < 2000]
    idxmissing = data == -200
    data = data[numpy.sum(idxmissing,1) == 0, :]
    idxmissing = data == -200
    print(data.shape)
    
    _, mixt = getArchetypes(data, dimensions)
    
    if mixt is None:
        print( "no archetypes", dimensions)
        #0/0
        return
    
    def normalize(data):
        mixtd = data
        mixtd[mixtd == 1] = mixtd[mixtd == 1] - 0.0000001
        mixtd[mixtd == 0] = mixtd[mixtd == 0] + 0.0000001
        mixtnorm = numpy.sum(mixtd, axis=1)
        mixtd = numpy.divide(mixtd, mixtnorm[:, None])
        return mixtd+0.0
    
    mixt = normalize(mixt)
    print(data.shape)
    featureTypes = ["continuous"] * mixt.shape[1]
    
    domains = [[0,1]] * mixt.shape[1]
    
    print(domains)
        
    @memory.cache
    def learn(data):
        spn = SPN.LearnStructure(data, featureTypes=featureTypes, row_split_method=Splitting.KmeansRDCRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                                 domains=domains,
                                 alpha=0.1,
                                 families = ['histogram'] * data.shape[1],
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 #min_instances_slice=int(data.shape[0]*0.01))
                                 min_instances_slice=200)
        return spn
    
    
    stats = Stats(name=dsname)
    
    for train, test, i in kfolded(mixt, 10):
        print(i)
        #dirichlet_alphas = getDirichlet(train)
        dirichlet_alphas = dirichlet.mle(train, method='meanprecision', maxiter=1000000)
        print("dirichlet done")
        ll = scipy.stats.dirichlet.logpdf(numpy.transpose(test), alpha=dirichlet_alphas)
        stats.add("DIRICHLET", Stats.LOG_LIKELIHOOD, numpy.sum(ll))
        stats.add("DIRICHLET", Stats.MEAN_LOG_LIKELIHOOD, numpy.mean(ll))
        
        spn = learn(train)
        print("spn done")
        ll = spn.root.eval(test)
        print(stats.add("SPN", Stats.LOG_LIKELIHOOD, numpy.sum(ll)))
        print(stats.add("SPN", Stats.MEAN_LOG_LIKELIHOOD, numpy.mean(ll)))

     
    stats.save("results/airpollution/"+ dsname + "-" + str(dimensions) + ".json")   

#with Pool(processes=8) as pool:
#    pool.map(getAirPollution, [3,5,10,20])
#for dimension in [3,5,10,20,50]:    
#    getAirPollution(dimension)


def getHydrochemLL():
    dsname, data, features = getHydrochem()

    print(data)
    print(data.shape)
    
    featureTypes = ["continuous"] * data.shape[1]
    
    domains = [[0,1]] * data.shape[1]

    print(domains)
    families = ['piecewise'] * data.shape[1] 
    #families = ['histogram'] * data.shape[1]                                                                                                                                                         
    #@memory.cache
    def learn(data, families, mininst, alpha, th):
        spn = SPN.LearnStructure(data, featureTypes=featureTypes, row_split_method=Splitting.Gower(), col_split_method=Splitting.RDCTest(threshold=th),                                               
                                 domains=domains,
                                 alpha=alpha,
                                 families = families,
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=mininst)
        return spn

    stats = Stats(name=dsname)

    alll = []
    for train, test, i in kfolded(data, 5):
        dirichlet_alphas = dirichlet.mle(train, method='meanprecision', maxiter=100000)
        ll = scipy.stats.dirichlet.logpdf(numpy.transpose(test), alpha=dirichlet_alphas)
        stats.add("DIRICHLET", Stats.LOG_LIKELIHOOD, numpy.sum(ll))
        stats.add("DIRICHLET", Stats.MEAN_LOG_LIKELIHOOD, numpy.mean(ll))

        spn = learn(train, families, 10, 0.1, 0.1)
        ll = spn.root.eval(test)
        alll.append(numpy.mean(ll))
        stats.add("SPN", Stats.LOG_LIKELIHOOD, numpy.sum(ll))
        stats.add("SPN", Stats.MEAN_LOG_LIKELIHOOD, numpy.mean(ll))

    print(numpy.mean(alll))
    stats.save("results/hydrochems/"+ dsname + ".json")


#getHydrochemLL()
#0/0



def computeNIPS(dimensions):
    dsname, data, features = getNips()
    
    lda = LatentDirichletAllocation(n_topics=dimensions, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    
    lda.fit(data)
    mixt = lda.transform(data)

    def normalize(data):
        mixtd = data
        mixtd[mixtd == 1] = mixtd[mixtd == 1] - 0.0000001
        mixtd[mixtd == 0] = mixtd[mixtd == 0] + 0.0000001
        mixtnorm = numpy.sum(mixtd, axis=1)
        mixtd = numpy.divide(mixtd, mixtnorm[:, None])
        return mixtd+0.0
    
    mixt = normalize(mixt)
    print(data.shape)
    featureTypes = ["continuous"] * mixt.shape[1]
    
    domains = [[0,1]] * mixt.shape[1]
    
    print(domains)
        
    @memory.cache
    def learn(data):
        spn = SPN.LearnStructure(data, featureTypes=featureTypes, row_split_method=Splitting.KmeansRDCRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                                 domains=domains,
                                 alpha=0.1,
                                 families = ['histogram'] * data.shape[1],
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=100)
        return spn
    
    
    stats = Stats(name=dsname)
    
    for train, test, i in kfolded(mixt, 10):
        print(i)
        #dirichlet_alphas = getDirichlet(train)
        dirichlet_alphas = dirichlet.mle(train, method='meanprecision', maxiter=1000000)
        print("dirichlet done")
        ll = scipy.stats.dirichlet.logpdf(numpy.transpose(test), alpha=dirichlet_alphas)
        stats.add("DIRICHLET", Stats.LOG_LIKELIHOOD, numpy.sum(ll))
        stats.add("DIRICHLET", Stats.MEAN_LOG_LIKELIHOOD, numpy.mean(ll))
        
        spn = learn(train)
        print("spn done")
        ll = spn.root.eval(test)
        print(stats.add("SPN", Stats.LOG_LIKELIHOOD, numpy.sum(ll)))
        print(stats.add("SPN", Stats.MEAN_LOG_LIKELIHOOD, numpy.mean(ll)))

     
    stats.save("results/nips/"+ dsname + "-" + str(dimensions) + ".json")   


#with Pool(processes=8) as pool:
#    pool.map(computeNIPS, [3,5,10,20,50])
    
#for dimension in [3,5,10,20,50]:  
computeNIPS(50)
0/0


def computeSimplexExperiment(dsname, data, dimensions, mixttype, min_instances_slice=700):
    if mixttype == "Archetype":
        _, mixt = getArchetypes(data, dimensions)
        if mixt is None:
            return ()
    elif mixttype == "LDA":
        lda = LatentDirichletAllocation(n_topics=dimensions, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    
        lda.fit(data)
        mixt = lda.transform(data)
    elif mixttype == "RandomSample":
        mixt = numpy.random.dirichlet((1,1,1), 20).transpose()
        print(mixt)
        0/0
        
    print(mixt.shape)
    
    def normalize(data):
        mixtd = data
        mixtd[mixtd == 1] = mixtd[mixtd == 1] - 0.0000001
        mixtd[mixtd == 0] = mixtd[mixtd == 0] + 0.0000001
        mixtnorm = numpy.sum(mixtd, axis=1)
        mixtd = numpy.divide(mixtd, mixtnorm[:, None])
        return mixtd+0.0
    
    mixt = normalize(mixt)
    mixt_train, mixt_test = train_test_split(mixt, test_size=0.30, random_state=42)


    numpy.savetxt("mixt_train.csv", mixt_train)
    numpy.savetxt("mixt_test.csv", mixt_test)
    #0/0

    featureTypes = ["continuous"] * mixt.shape[1]
    
    domains = []
    for i, ft in enumerate(featureTypes):
        if ft == "continuous":
            r = (0.0, 1.0)
            fd = estimate_continuous_domain(mixt, i, range=r, binning_method=400)
            domain = numpy.array(sorted(fd.keys()))
        else:
            domain = numpy.unique(data[:, i])

        domains.append(domain)
    
    dirichlet_alphas = dirichlet.mle(mixt_train, method='meanprecision', maxiter=100000)

    #@memory.cache
    def learn(data):
        spn = SPN.LearnStructure(data, featureTypes=featureTypes, row_split_method=Splitting.KmeansRDCRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                                 domains=domains,
                                 alpha=0.1,
                                 families = ['histogram'] * data.shape[1],
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=min_instances_slice)
        return spn
    #for the good pdf it was 700
    
    
    spn = learn(mixt_train)
    print(spn)
    def spnpdf(data):
        data = data.reshape(-1, mixt.shape[1])
        res = spn.root.eval(normalize(data))[0]
        return res
    
    print(dirichlet_alphas)
    
    def plotDirichlet(data):
        data = data.reshape(-1, mixt.shape[1])
        try:
            result = scipy.stats.dirichlet.logpdf(numpy.transpose(normalize(data)), alpha=dirichlet_alphas)
        except:
            print(normalize(data))
            print(normalize(data)*1.0)
            print(normalize(data)+1)
            print(normalize(data)+0)
            0/0
        return result
    
    df_train = pandas.DataFrame()
    df_test = pandas.DataFrame()
    
    dtrain_fit = scipy.stats.dirichlet.logpdf(numpy.transpose(mixt_train), alpha=dirichlet_alphas)
    dtest_fit = scipy.stats.dirichlet.logpdf(numpy.transpose(mixt_test), alpha=dirichlet_alphas)
    df_train["dirichlet_train"] = dtrain_fit
    df_test["dirichlet_test"] = dtest_fit
    
    spn_train_fit = spn.root.eval(mixt_train)
    spn_test_fit = spn.root.eval(mixt_test)
    df_train["spn_train"] = spn_train_fit
    df_test["spn_test"] = spn_test_fit
    

    
    if dimensions == 3:
        xy_train = cartesian(mixt_train)
        xy_test = cartesian(mixt_test)
        
        filename = 'plots/%s_%s.pdf' % (dsname, mixttype)
        try:
            import os
            os.remove(filename)
        except OSError:
            pass
        pp = PdfPages(filename)
        
        markersize = 1.0
        # all
#         fig = plt.figure()
#         plt.title("dirichlet, original points")
#         draw_pdf_contours_func2(plotDirichlet, vmin=-2, vmax=12)
#         #draw_pdf_contours_func2(xy_all[:, 0], xy_all[:, 1],plotDirichlet)
#         plt.plot(xy_all[:, 0], xy_all[:, 1], 'ro', markersize=markersize)
#         plt.colorbar()
#         pp.savefig(fig)
        # train
        fig = plt.figure()
        plt.title("Dirichlet, train points")
        draw_pdf_contours_func2(plotDirichlet, vmin=-2, vmax=12)
        #draw_pdf_contours_func2(xy_all[:, 0], xy_all[:, 1],plotDirichlet)
        plt.plot(xy_train[:, 0], xy_train[:, 1], 'ro', markersize=markersize)
        #plt.colorbar()
        fig.tight_layout()
        pp.savefig(fig)
        
        # test
        fig = plt.figure()
        plt.title("Dirichlet, test points")
        draw_pdf_contours_func2(plotDirichlet, vmin=-2, vmax=12)
        #draw_pdf_contours_func2(xy_all[:, 0], xy_all[:, 1],plotDirichlet)
        plt.plot(xy_test[:, 0], xy_test[:, 1], 'ro', markersize=markersize)
        #plt.colorbar()
        fig.tight_layout()
        pp.savefig(fig)
    
        # all
#         fig = plt.figure()
#         plt.title("spn, original points")
#         draw_pdf_contours_func2(spnpdf, vmin=-2, vmax=12)
#         #draw_pdf_contours_func2(xy_all[:, 0], xy_all[:, 1],spnpdf)
# 
#         plt.plot(xy_all[:, 0], xy_all[:, 1], 'ro', markersize=markersize)
#         plt.colorbar()
#         pp.savefig(fig)
        
        # train
        fig = plt.figure()
        plt.title("SPN, train points")
        draw_pdf_contours_func2(spnpdf, vmin=-2, vmax=12)
        #draw_pdf_contours_func2(xy_all[:, 0], xy_all[:, 1],spnpdf)
        plt.plot(xy_train[:, 0], xy_train[:, 1], 'ro', markersize=markersize)
        #plt.colorbar()
        fig.tight_layout()
        pp.savefig(fig)
        
        # test
        fig = plt.figure()
        plt.title("SPN, test points")
        draw_pdf_contours_func2(spnpdf, vmin=-2, vmax=12)
        #draw_pdf_contours_func2(xy_all[:, 0], xy_all[:, 1],spnpdf)
        plt.plot(xy_test[:, 0], xy_test[:, 1], 'ro', markersize=markersize)
        #plt.colorbar()
        fig.tight_layout()
        pp.savefig(fig)
        pp.close()
    
    return ("name", dsname, "size", data.shape, "type", mixttype, "dims", dimensions,
            "spn_train_LL", numpy.mean(spn_train_fit), "dir_train_LL", numpy.mean(dtrain_fit),
            "spn_test_LL", numpy.mean(spn_test_fit), "dir_test_LL", numpy.mean(dtest_fit) ,
            "spn_#_sum_nodes", spn.n_sum_nodes(), "spn_#_prod_nodes", spn.n_prod_nodes(), "spn_#_layers", spn.n_layers()
            )

    
#     fig = plt.figure(1, figsize=(9, 6))
#     ax = fig.add_subplot(111)
#     data_to_plot = [dtrain_fit, spn_train_fit]
#     bp = ax.boxplot(data_to_plot)
#     
#     plt.show()    

#     import seaborn as sns
#  
#     sns.set_style("whitegrid")
#     ax = sns.boxplot(data=df_train)
#     ax = sns.swarmplot(data=df_train, color=".15")
#     plt.show()
#      
#     ax = sns.boxplot(data=df_test)
#     ax = sns.swarmplot(data=df_test, color=".15")
    
    

dsname, data, features = getNips()
results = []

#print(computeSimplexExperiment("Nips", data, 3, "RandomSample"))



print(computeSimplexExperiment("Nips", data, 3, "LDA", 1000)) #makes plot
0/0


for dimension in [5,10,20,50]:
    for mixttype in ["Archetype", "LDA"]:
        results.append([computeSimplexExperiment("Nips", data, dimension, mixttype)])

#data, features, times, hours = getTraffic()
#for dimension in [5,10,20,50]:
#    for mixttype in ["Archetype", "LDA"]:
#        results.append([computeSimplexExperiment("Traffic", data, dimension, mixttype)])


print(results)

0/0



def test6(data):
    print(data.shape)
    _, mixt = getArchetypes(data, 3)
    
    def normalize(data):
        mixtd = data
        mixtd[mixtd == 1] = mixtd[mixtd == 1] - 0.0000001
        mixtd[mixtd == 0] = mixtd[mixtd == 0] + 0.0000001
        mixtnorm = numpy.sum(mixtd, axis=1)
        mixtd = numpy.divide(mixtd, mixtnorm[:, None])
        return mixtd
    
    mixt = normalize(mixt)
    
    dirichlet_alphas = dirichlet.mle(mixt, method='meanprecision', maxiter=100000)
    
    featureTypes = ["continuous"] * mixt.shape[1]
    
    domains = []
    for i, ft in enumerate(featureTypes):
        if ft == "continuous":
            r = (0.0, 1.0)
            fd = estimate_continuous_domain(mixt, i, range=r, binning_method=400)
            domain = numpy.array(sorted(fd.keys()))
        else:
            domain = numpy.unique(data[:, i])

        domains.append(domain)
    print(domains)

    @memory.cache
    def learn(data):
        spn = SPN.LearnStructure(data, featureTypes=featureTypes, row_split_method=Splitting.Gower(), col_split_method=Splitting.RDCTest(threshold=0.3),
                                 domains=domains,
                                 alpha=1,
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=50)
        return spn
    
    spn = learn(mixt)
    print(spn)
    
    spn_samples = numpy.zeros((data.shape[0], 3))/0
    a,spn_samples = spn.root.sample(spn_samples)
    
    spn_samples = normalize(spn_samples)
    
    
    
    #dtrain_fit = scipy.stats.dirichlet.logpdf(numpy.transpose(mixt_train), alpha=dirichlet_alphas)
    def plotDirichlet(data):
        data = data.reshape(-1, mixt.shape[1])
        result = scipy.stats.dirichlet.logpdf(numpy.transpose(normalize(data)), alpha=dirichlet_alphas)
        return result
    
    def spnpdf(data):
        data = data.reshape(-1, mixt.shape[1])
        res = spn.root.eval(normalize(data))[0]
        return res
    
    xy_all = cartesian(mixt)
    
    
    filename = 'plots/dirichlet_mle.pdf'
    try:
        import os
        os.remove(filename)
    except OSError:
        pass
    pp = PdfPages(filename)
    
    # all
    fig = plt.figure()
    draw_pdf_contours_func(plotDirichlet)
    plt.title("dirichlet trained on all, original points")
    plt.plot(xy_all[:, 0], xy_all[:, 1], 'ro', markersize=markersize)
    plt.colorbar()
    pp.savefig(fig)
    
    numpy.random.seed(17)
    mixt_samples = numpy.random.dirichlet(dirichlet_alphas, data.shape[0])
    print(dirichlet_alphas)
    xy_samples = cartesian(mixt_samples)
    
    
    fig = plt.figure()
    draw_pdf_contours_func(plotDirichlet)
    plt.title("dirichlet trained on all, sampled points")
    plt.plot(xy_samples[:, 0], xy_samples[:, 1], 'ro', markersize=markersize)
    plt.colorbar()
    pp.savefig(fig)
    
    xy_spn_samples = cartesian(spn_samples)
    fig = plt.figure()
    draw_pdf_contours_func(spnpdf)
    plt.title("spn trained on all, original points")
    plt.plot(xy_all[:, 0], xy_all[:, 1], 'ro', markersize=markersize)
    plt.colorbar()
    pp.savefig(fig)
    
    
    xy_spn_samples = cartesian(spn_samples)
    fig = plt.figure()
    draw_pdf_contours_func(spnpdf)
    plt.title("spn trained on all, sampled points")
    plt.plot(xy_spn_samples[:, 0], xy_spn_samples[:, 1], 'ro', markersize=markersize)
    plt.colorbar()
    pp.savefig(fig)
    
    
    
    pp.close()

#test6(data)
#0/0

def other():
    corners = polycorners(mixt.shape[1])
    print(corners)

    def plotSimplex(func):
        x = y = numpy.arange(0.0, 1.01, 0.01)
        XY = numpy.dstack(numpy.meshgrid(x, y)).reshape(-1, 2)
        barcoords = cart2bary(corners, XY)
        idx = numpy.sum(barcoords < 0, axis=1) == 0
        Z = numpy.zeros((XY.shape[0], 1))
        Z[idx, ] = func(barcoords[idx, ]).reshape(-1, 1)

        fig = plt.figure(1, figsize=(6, 6))
        ax = fig.add_subplot(111)
        
        # poly = Polygon(corners, closed=True, fill=False, alpha=0.5)
        # ax.add_patch(poly)
        
        
        x = XY[idx, 0]
        y = XY[idx, 1]
        z = Z[idx, 0]
        
        import matplotlib.mlab as ml
        np = 100
        xi = numpy.linspace(0, 1, np)
        yi = numpy.linspace(0, 1, np)
        zi = ml.griddata(x, y, z, xi, yi, interp="linear")
        
        plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
        plt.pcolormesh(xi, yi, zi, cmap=plt.get_cmap('rainbow'))
        plt.show()
        
        0 / 0
        
        # plt.contourf(x, y, z, 20, linewidths = 0.5)
        plt.contour(x, y, z, 15, linewidths=0.5, colors='k')

        plt.pcolormesh(x, y, z)
        
    
    # plotSimplex(plotDirichlet)      
    # plt.show()
    # 0/0
    

def test4(data, features):
    n_topics = 3
    
    featureNames = features + ["%s%s" % x for x in zip(["topic"] * n_topics, range(n_topics))]
    
    f = numpy.array(featureNames)
    
    
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    
    numpy.random.seed(17)
    trainIdx = numpy.random.choice([True, False], size=(data.shape[0],), p=[2. / 3, 1. / 3])
    testIdx = numpy.logical_not(trainIdx)
    
    train = data[trainIdx, :]
    test = data[testIdx, :]
    
    lda.fit(train)
    
    topicstrain = lda.transform(train)
    topicstest = lda.transform(test)
    
    
    maxtrain = numpy.argmax(topicstrain, axis=1)
    topicstrain = numpy.zeros_like(topicstrain)
    for i, c in enumerate(maxtrain):
        topicstrain[i, c] = 1
        
    maxtest = numpy.argmax(topicstest, axis=1)
    topicstest = numpy.zeros_like(topicstest)
    for i, c in enumerate(maxtest):
        topicstest[i, c] = 1
    
    
    
    testMPE = numpy.zeros((test.shape[0], len(featureNames)))
    testMPE = testMPE / 0
    testMPE[:, numpy.arange(test.shape[1])] = test
    
    print("LEARNING SPN")
    
    traintopics = numpy.hstack((train, topicstrain))
    
    # print(testMPE[:,[99, 100,101,102]])

    # print(traintopics[:,[99, 100,101,102]])

    print(featureNames)
    
    featureTypes = ["discrete"] * train.shape[1] + ["continuous"] * topicstrain.shape[1]
    
    domains = []
    for i, ft in enumerate(featureTypes):
        if ft == "continuous":
            r = (0.0, 1.0)
            fd = estimate_continuous_domain(traintopics, i, range=r, binning_method=20)
            domain = numpy.array(sorted(fd.keys()))
        else:
            domain = numpy.unique(data[:, i])

        # print(i, ft, domain)
        domains.append(domain)

        
    memory = Memory(cachedir="/tmp/test4", verbose=0, compress=9)


    @memory.cache
    def learn():
        spn = SPN.LearnStructure(traintopics, featureTypes=featureTypes, row_split_method=Splitting.Gower(), col_split_method=Splitting.RDCTest(threshold=0.1, linear=True),
                                featureNames=featureNames,
                                domains=domains,
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=100)
        return spn
    
    spn = learn()

    spn.root.validate()
    
    prodNodes = spn.get_nodes_by_type(ProductNode)

    for pn in prodNodes:
        leaves = pn.get_leaves()
        words = set()
        for leaf in leaves:
            # assuming pwl node:
            _x = numpy.argmax(leaf.y_range)
            max_x = leaf.x_range[_x]
            if max_x < 1.0:
                continue
            
            words.add(featureNames[leaf.featureIdx])
        # ll = pn.eval()
        if len(words) < 4:
            continue
        
        print(words)
        

    
    logs, topicsmpe = spn.root.mpe_eval(testMPE)
    
    print(spn.get_leaves())
    
    print(topicsmpe.shape)
    
    print(topicsmpe[:, [100, 101, 102]])
    
    
    maxmpe = numpy.argmax(topicsmpe[:, [100, 101, 102]], axis=1)
    topicsmpe = numpy.zeros_like(topicsmpe[:, [100, 101, 102]])
    for i, c in enumerate(maxmpe):
        topicsmpe[i, c] = 1
        
        
    print(topicstest)
    print(topicsmpe)

    print(topicstest - topicsmpe)
    correct = numpy.sum(numpy.abs(topicstest - topicsmpe), axis=1) == 0

    print("correct", numpy.sum(correct))
    print("incorrect", topicsmpe.shape[0] - numpy.sum(correct))
    
    print_top_words(lda, features, 10)
    
    

    

def test3(data, features):
    
    
    f = numpy.array(features)
    
    
    
    
    lda = LatentDirichletAllocation(n_topics=3, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    
    
    lda.fit(data)
    
    topics = lda.transform(data)
    
    
    print("LEARNING SPN")
    memory = Memory(cachedir="/tmp/test3", verbose=0, compress=9)

    @memory.cache
    def learn():
        spn = SPN.LearnStructure(data, featureTypes=["discrete"] * data.shape[1], row_split_method=Splitting.Gower(), col_split_method=Splitting.RDCTest(threshold=0.3, linear=True),
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=200)
    

        return spn
    
    spn = learn()
    spn.root.validate()
    
    prodNodes = spn.get_nodes_by_type(ProductNode)

    for pn in prodNodes:
        leaves = pn.get_leaves()
        words = set()
        for leaf in leaves:
            # assuming pwl node:
            _x = numpy.argmax(leaf.y_range)
            max_x = leaf.x_range[_x]
            if max_x < 1.0:
                continue
            
            words.add(features[leaf.featureIdx])
        # ll = pn.eval()
        if len(words) < 4:
            continue
        
        print(words)
        

    
    0 / 0
    prodNodes = spn.get_nodes_by_type(ProductNode)
    
    
    pnll = numpy.zeros((data.shape[0], len(prodNodes)))
    
    for i, pn in enumerate(prodNodes):
        pnll[:, i] = numpy.exp(pn.eval(data))
        
        
    
    for i in range(topics.shape[1]):
        tmax = 0
        pnmax = None
        
        for j in range(len(prodNodes)):
            # rdcval = rdcdcor(topics[:,i], pnll[:,j])
            bic = numpy.log(data.shape[0]) * len(prodNodes[j].scope) - 2.0 * numpy.log(numpy.sum(pnll[:, j]))
            if bic > tmax:
                tmax = bic
                pnmax = prodNodes[j]
    
    
        print("spn topic")
        print(pnmax.scope)
        print(f[list(pnmax.scope)])
    
    print()
    print_top_words(lda, features, 10)
    
    
        
    # print(pnll)
    
    
    # print(spn)

# test3(data, features)



def test2(data, features):
    arc, mixt = getArchetypes(data, 3)
    
    print(mixt)
    
    0 / 0
    
    
    spn = SPN.LearnStructure(mixt, featureTypes=["continuous"] * mixt.shape[1], row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=100)
    
    
    
    

def test1(data, features):
    
    data = data[:, 1:20]
    features = features[0:data.shape[1]]

    arcs, mixt = getArchetypes(data, 3)
    
    nrfolds = 10
    
    
    stats = Stats(name=dsname)
    
    for train, test, i in kfolded(mixt, nrfolds):
        c = Chrono().start()
        spn = SPN.LearnStructure(train, featureTypes=["continuous"] * train.shape[1], row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=100)
        c.end()
        
        spn.root.validate()
        ll = numpy.mean(spn.root.eval(test))
        
        print(ll)
        
        stats.add("HSPN", Stats.LOG_LIKELIHOOD, ll)
        stats.add("HSPN", Stats.TIME, c.elapsed())
        
        stats.save("stats_" + dsname + ".json")
    
    print(arcs)
    
    
    
