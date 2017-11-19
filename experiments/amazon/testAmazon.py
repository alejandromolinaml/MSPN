'''
Created on 3 Jun 2017

@author: alejomc



'''
import codecs
import itertools
from multiprocessing.pool import Pool

from joblib.memory import Memory
import numpy
from sklearn import linear_model, metrics, cross_validation
from sklearn.base import BaseEstimator
from sklearn.decomposition.online_lda import LatentDirichletAllocation
from sklearn.linear_model.logistic import LogisticRegression

from mlutils.test import kfolded
import multiprocessing  as mp
from pdn.independenceptest import getIndependentGroups, getIndependentRDCGroups
from tfspn.SPN import SPN, Splitting
from nltk.tbl import feature


memory = Memory(cachedir="/tmp/amazon", verbose=0, compress=9)

# https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
def readFile(fname, domain):
    docs = []
    with codecs.open(fname, "r", "utf-8") as f:
        for line in f:
            cols = line.strip().split(" ")
            
            kv = {}
            for c in cols:
                k, v = c.split(":")
                if k == "#label#":
                    if v == "positive":
                        v = 1
                    else:
                        v = 0
                    
                kv[k] = int(v)
            kv["#domain#"] = domain
            docs.append(kv)
            
    return docs

def getFreqs(lst):
    corpusFreq = {}
    
    for doc in lst:
        for k, v in doc.items():
            if k not in corpusFreq:
                corpusFreq[k] = 0
            if k == "#label#":
                corpusFreq[k] += 1
            else:
                corpusFreq[k] += v
            
    return corpusFreq

def getBOW(docslist, informativeFeatures):
    bow = numpy.zeros((len(docslist), len(informativeFeatures)))
    
    for i, doc in enumerate(docslist):
        for j, k in enumerate(informativeFeatures):
            if k in doc:
                bow[i, j] = doc[k]
    
    return bow

def getInformativeFeatures(corpusFreq, fgt=5):
    informativeFeatures = set([k for (k, v) in corpusFreq.items() if v > fgt])
    
    informativeFeatures.remove("#domain#")
    informativeFeatures.remove("#label#")
    
    informativeFeatures = list(informativeFeatures)

    informativeFeatures.append("#domain#")
    informativeFeatures.append("#label#")
    return informativeFeatures

@memory.cache
def getAmazonData(fgt=5):
    booksneg = readFile("processed_acl/books/negative.review", 1)
    bookspos = readFile("processed_acl/books/positive.review", 1)
    dvdneg = readFile("processed_acl/dvd/negative.review", 2)
    dvdpos = readFile("processed_acl/dvd/positive.review", 2)
    electronicsneg = readFile("processed_acl/electronics/negative.review", 3)
    electronicspos = readFile("processed_acl/electronics/positive.review", 3)
    kitchenneg = readFile("processed_acl/kitchen/negative.review", 4)
    kitchenpos = readFile("processed_acl/kitchen/positive.review", 4)
    
    allc = []
    allc.extend(booksneg)
    allc.extend(bookspos)
    allc.extend(dvdneg)
    allc.extend(dvdpos)
    allc.extend(electronicsneg)
    allc.extend(electronicspos)
    allc.extend(kitchenneg)
    allc.extend(kitchenpos)
    
    print(len(allc))
    
    corpusFreq = getFreqs(allc)
    
    informativeFeatures = getInformativeFeatures(corpusFreq, fgt=fgt)
    
    
    print(len(informativeFeatures))
    
    bow = getBOW(allc, informativeFeatures)
    return bow

@memory.cache
def learn(data, featureTypes, families, domains, min_instances_slice, alpha=0.1):
    spn = SPN.LearnStructure(data, alpha=alpha, featureTypes=featureTypes, row_split_method=Splitting.KmeansRDCRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                             domains=domains,
                             families=families,
                             min_instances_slice=min_instances_slice)
    return spn

@memory.cache
def getLDA(data, dimensions):
    lda = LatentDirichletAllocation(n_topics=dimensions, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    
    lda.fit(data)
    return lda.transform(data)

min_instances_slice = 200

def getBOW2SA(data):
    xtrain = data[:, 0:10]
    ytrain = data[:, -1].reshape(data.shape[0], -1)
    
    featureTypes = ["discrete"] * xtrain.shape[1] + ["categorical"] * ytrain.shape[1]
    families = ["isotonic"] * xtrain.shape[1] + ["isotonic"] * ytrain.shape[1]
    
    return xtrain, None, ytrain, featureTypes, families

def getBOW_CAT2SA(data):
    bow = data[:, 0:10]
    cat = data[:, -2].reshape(-1, 1)
    ytrain = data[:, -1].reshape(data.shape[0], -1)

    featureTypes = ["discrete"] * bow.shape[1] + ["discrete"] * cat.shape[1] + ["categorical"] * ytrain.shape[1]
    families = ["isotonic"] * len(featureTypes)
    
    return bow, cat, ytrain, featureTypes, families



def getBOW_LDA2SA(data):
    bow = data[:, 0:10]
    y = data[:, -1].reshape(data.shape[0], -1)

    dimensions = 10
    mixt = getLDA(data[:, 0:(data.shape[1] - 2)], dimensions)
    

    featureTypes = ["discrete"] * bow.shape[1] + ["continuous"] * mixt.shape[1] + ["categorical"] * y.shape[1]
    families = ["isotonic"] * len(featureTypes)
    
    return bow, mixt, y, featureTypes, families


def getBOW_CAT_LDA2SA(data):
    bow = data[:, 0:10]
    y = data[:, -1].reshape(data.shape[0], -1)

    dimensions = 10
    mixt = getLDA(data[:, 0:(data.shape[1] - 2)], dimensions)
    
    categories = data[:, -2].reshape(-1, 1)

    featureTypes = ["discrete"] * bow.shape[1] + ["continuous"] * mixt.shape[1] + ["categorical"] * categories.shape[1] + ["categorical"] * y.shape[1]
    families = ["isotonic"] * len(featureTypes)
    
    return bow, numpy.hstack((mixt, categories)), y, featureTypes, families


def doCV(funct, datafunc, data, folds=10, ncores=4):
    itertasks = [(datafunc, train, test, i) for train, test, i in kfolded(data, folds)]
    if ncores > 1:
        ctx = mp.get_context('forkserver')
        with ctx.Pool(processes=ncores) as pool:
            return numpy.vstack((pool.starmap(funct, itertasks)))
    else:
        return numpy.vstack((itertools.starmap(funct, itertasks)))
    
def LRfitpredict(datafunc, train, test, i):
    xtrain, _, ytrain, _, _ = datafunc(train)
    xtest, _, _, _, _ = datafunc(test)
    lr = LogisticRegression(random_state=17)
    lr.fit(xtrain, ytrain[:, 0])
    return lr.predict(xtest).reshape(-1, 1)

def SPNfitpredict(datafunc, train, test, i):
    xtrain, xpriv, ytrain, featureTypes, families = datafunc(train)
    xtest, _, _, _, _ = datafunc(test)

    allx = numpy.vstack((xtrain, xtest))
    domains = [ numpy.unique(allx[:, i]) for i in range(allx.shape[1]) ]
    
    if xpriv is None:
        data = numpy.hstack((xtrain, ytrain))
    else:
        data = numpy.hstack((xtrain, xpriv, ytrain))
        domains += [ numpy.unique(xpriv[:, i]) for i in range(xpriv.shape[1]) ]
        
    domains += [ numpy.unique(ytrain[:, i]) for i in range(ytrain.shape[1]) ]

    spn = learn(data, featureTypes, families, domains=domains, min_instances_slice=min_instances_slice, alpha=0.1)
    
    print("n_sum_nodes", spn.n_sum_nodes(), "n_prod_nodes", spn.n_prod_nodes(), "n_layers", spn.n_layers())
    
    mpe = numpy.zeros((xtest.shape[0], len(featureTypes)))
    mpe /= 0
    mpe[:, 0:xtest.shape[1]] = xtest
    _, vals = spn.root.mpe_eval(mpe)
    return vals[:, xtest.shape[1]:].reshape(xtest.shape[0], -1)

if __name__ == '__main__':
    folds = 10
    data = getAmazonData(fgt=1200)
    print(data.shape)
    #0/0
    
    numpy.random.seed(1)
    numpy.random.shuffle(data)
    
    _, _, target, _, _ = getBOW2SA(data)
    
    lrpreds = doCV(LRfitpredict, getBOW2SA, data, folds=folds, ncores=4)
    print("LR")
    print(metrics.accuracy_score(target, lrpreds))
    print(metrics.classification_report(target, lrpreds))
    
    spnpreds = doCV(SPNfitpredict, getBOW2SA, data, folds=folds, ncores=4)
    print("SPN BOW -> SA")
    print(metrics.accuracy_score(target, spnpreds))
    print(metrics.classification_report(target, spnpreds))
    
    spnpreds = doCV(SPNfitpredict, getBOW_CAT2SA, data, folds=folds, ncores=4)
    print("SPN BOW + CAT -> CAT + SA")
    print(metrics.accuracy_score(target, spnpreds[:, -1]))
    print(metrics.classification_report(target, spnpreds[:, -1]))
    
    spnpreds = doCV(SPNfitpredict, getBOW_LDA2SA, data, folds=folds, ncores=4)
    print("SPN BOW + LDA -> LDA + SA")
    print(metrics.accuracy_score(target, spnpreds[:, -1]))
    print(metrics.classification_report(target, spnpreds[:, -1]))
    
    spnpreds = doCV(SPNfitpredict, getBOW_CAT_LDA2SA, data, folds=folds, ncores=4)
    print("SPN BOW + CAT + LDA -> CAT + LDA + SA")
    print(metrics.accuracy_score(target, spnpreds[:, -1]))
    print(metrics.classification_report(target, spnpreds[:, -1]))
    

    
    
    
    

