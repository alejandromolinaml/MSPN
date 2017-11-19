'''
Created on Feb 15, 2016

@author: molina
'''


import math

import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.frame import H2OFrame
import numpy
import re

from mlutils.h2outils import numpytoordereddict
from mlutils.statistics import logpoissonpmf

import h2o
h2o.init()
h2o.no_progress()


class GBMPDN:
    
    def __init__(self, data, features, families="poisson", max_depth=10, iterations=1):
        self.data = data
        self.nD = data.shape[0]
        self.nF = data.shape[1]
        
        self.config = {"max_depth":max_depth, "iterations":iterations}
        
        self.vars = numpy.var(data, 0)
        self.means = numpy.mean(data, 0)
        
        assert self.nF == len(features)

        self.features = features

        if isinstance(families, str):
            families = [families] * self.nF

        updata = H2OFrame(numpytoordereddict(self.data, self.features))
        
        self.models = {}
        for i, feature in enumerate(self.features):
            if self.vars[i] == 0:
                continue
            
            self.models[feature] = H2OGradientBoostingEstimator(distribution="poisson", ntrees=iterations, max_depth=max_depth)
            self.models[feature].train(x=[f for f in self.features if f != feature], y=feature, training_frame=updata)
    
    
    def getLambdas(self, documents):
        uptest = H2OFrame(numpytoordereddict(documents, self.features))
        
        result = numpy.zeros(documents.shape)
        
        for i, feature in enumerate(self.features):
            if self.vars[i] == 0:
                lmbda = self.means[i]
                if lmbda == 0:
                    lmbda = 0.01
                result[:, i] = lmbda
                continue
            predframe = self.models[feature].predict(uptest)
            result[:, i] = list(map(float, sum(h2o.as_list(predframe, use_pandas=False) , [])[1:]))
           
        return result
    
    def getLogLikelihood(self, documents, debug=False):
        ll = 0
        
        lambdas = self.getLambdas(documents)
        
        for i in range(documents.shape[0]):
            for j in range(documents.shape[1]):
                ll += logpoissonpmf(documents[i, j], lambdas[i, j])
        
        return ll
    
    def perplexity(self, input):
        words = numpy.sum(input)
        ll = self.getLogLikelihood(input)
        pwb = ll / words
        return (pwb, numpy.exp2(-pwb), words, ll)
    
    
    def complete(self, test, nsamples=3000, burnin=200):
        data = numpy.copy(test)
        idxs = list(map(int, numpy.where(numpy.where(test, 0, 1) == 1)[0]))
        assert(len(idxs) > 0)
        
        if len(idxs) == 1:
            burnin = 0
            nsamples = 1
        
        samples = numpy.zeros((burnin + nsamples, len(idxs)))
        data[idxs] = numpy.floor(numpy.exp(self.initial_model[idxs]))
        for i in range(0, samples.shape[0]):
            k = i % len(idxs)
            j = idxs[k]
            data[j] = numpy.floor(self.getLambdas(data, [j])[j])
            samples[i, ] = data[idxs]
        # print(samples)
        data = numpy.copy(test)  
        data[idxs] = numpy.floor(numpy.mean(samples[burnin:, ], axis=0))
        return data
    
    def size(self):
        nodes = 0
        for i, feature in enumerate(self.features):
            if self.vars[i] == 0:
                continue
            
            model = self.models[feature]
            
            modelstr = h2o.connection.H2OConnection.get("Models.java/"+model.model_id).content.decode('utf-8')
            nodes += len(re.findall('\\?', modelstr))
            
        return nodes
    
#    def __repr__(self):
#        return "PDN: %s" % (self.getLogLikelihood(self.data))
    
    
    @staticmethod
    def pdnClustering(data, nM=2, boostlayers=2, maxIters=100, max_depth=3):
        nD = data.shape[0]
        nF = data.shape[1]
        indices = numpy.random.permutation(nD)
        # indices = numpy.random.random_integers(0,100,24)
        # indices = numpy.asarray(range(0,nD))
        splits = numpy.array_split(indices, nM)
        partitions = list(map(lambda idx: data[idx], splits))
        
        prevZ = numpy.zeros(nD, dtype=numpy.int)
        for iter in range(0, maxIters):
            pdns = []
            for i in range(0, nM):
                # print(partitions[i])
                dset = partitions[i]
                # print(dset.shape)
                pdn = GBMPDN(dset, max_depth=max_depth)
                for bl in range(0, boostlayers):
                    pdn.addBoostIteration()
                pdns.append(pdn)
            
            z = numpy.zeros(nD, dtype=numpy.int)
            for d in range(0, nD):
                doc = data[d, :]
                
                lls = list(map(lambda pdn: pdn.getLogLikelihood(doc), pdns))
                # print(lls)
                value = max(lls)
                index = lls.index(value)
                z[d] = index
            
            
            # print(sum(z == prevZ))
            
            if all(z == prevZ):
                break
            else:
                prevZ = z
                
            # print(iter)
            partitions = []
            for partition in range(0, len(numpy.unique(z))):
                # print(partition)
                # print(z == partition)
                partitions.append(data[z == partition, :])
                
            if len(partitions) < nM:
                return z
        
        return z    
    
if __name__ == '__main__':
    numpy.random.seed(42)
    gen = numpy.random.poisson(5, 1000)
    gen = numpy.hstack((gen,numpy.random.poisson(10, 1000)))
    gen = numpy.hstack((gen,numpy.random.poisson(15, 1000)))
    gen = numpy.hstack((gen,numpy.random.poisson(20, 1000)))
    
    
    #gen = numpy.hstack((gen, numpy.random.poisson(25, 100)))
    
    data = numpy.vstack((gen,numpy.random.permutation(gen),numpy.random.permutation(gen),numpy.random.permutation(gen)))
    
    data = numpy.transpose(data)
    print(data.shape)
    
    pdn = GBMPDN(data, ["V1","V2", "V3", "V4"], iterations=3)
    print(pdn.size())