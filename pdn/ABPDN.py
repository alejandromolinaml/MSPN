'''
Created on Feb 15, 2016

@author: molina
'''
from io import StringIO
from math import floor
import math

import numpy
from scipy.stats._discrete_distns import poisson
from sklearn import tree
from sklearn.tree.tree import DecisionTreeRegressor

class ABPDN:
    
    def __init__(self, data, max_depth=20, iterations=0):
        self.data = data
        self.max_depth = max_depth
        self.nD = data.shape[0]
        self.nF = data.shape[1]

        mean = numpy.mean(data, axis=0)
        mean[mean == 0] = 0.1
        self.initial_model = numpy.log(mean)
        self.trees = []
        
        for i in range(iterations):
            self.addBoostIteration()
    
#     def updateLambdas(self):
#         result = numpy.zeros(self.intermediate_predictions[0].shape)
#         
#         for i in range(0, len(self.intermediate_predictions)):
#             result += self.intermediate_predictions[i]
#         
#         self.lambdas = numpy.exp(result)
    
    def regressionValues(self):
        
        correction = numpy.zeros(self.data.shape)
        correction[self.data == 0] = 0.01
        
        lambdas = numpy.zeros(self.data.shape)
        for d in range(0, self.nD):
            lambdas[d, ] = self.getLambdas(self.data[d])
        
        return numpy.log(self.data + correction) - numpy.log(lambdas)


    def plotTrees(self, fname):
        import pydotplus
        
        for i, dtree in enumerate(self.trees):
            dotfile = StringIO()
            tree.export_graphviz(dtree[0], out_file=dotfile)
            g = pydotplus.graph_from_dot_data(dotfile.getvalue())
            g.write_png(fname.replace("*",str(i)))

    def addBoostIteration(self):
        rv = self.regressionValues()
        trees = []
        mask = numpy.array([True] * self.nF)
        for i in range(0, self.nF):
            mask[:] = True
            mask[i] = False
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(self.data[:, mask], rv[:, i])
            # newpsis[:, i] = tree.predict(self.data[:, mask])
            trees.append(tree)
        self.trees.append(trees)
        # self.intermediate_predictions.append(newpsis)
        # self.updateLambdas()
    
    def getLambdas(self, document, features=None):
        result = numpy.copy(self.initial_model)  # we start with the first constant
        
        mask = numpy.array([True] * self.nF)
        for tree in self.trees:
            if features:
                r = features
            else:
                r = range(0,self.nF)
            for i in r:
                mask[:] = True
                mask[i] = False
                result[i] += tree[i].predict(document[mask].reshape(1, -1))
        result = numpy.exp(result)
        # print(result)
        return result
    
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
        #print(samples)
        data = numpy.copy(test)  
        data[idxs] = numpy.floor(numpy.mean(samples[burnin:,], axis=0))
        return data
            
    
    def getLogLikelihood(self, documents, debug=False):
        ll = 0
        
        if documents.ndim > 1:
            for i in range(0, documents.shape[0]):
                ll += self.getLogLikelihood(documents[i], debug)
        else:
            # it's only one
            document = documents
            lambdas = self.getLambdas(document)
            if debug:
                print(document)
                print(numpy.round(lambdas, 0))
            for j in range(0, self.nF):
                pmfval = poisson.pmf(document[j], lambdas[j])
                if pmfval == 0.0:
                    pmfval = 0.000001
                    #pmfval = numpy.finfo(float).eps
                ll += math.log(pmfval)
        return ll

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
                pdn = ABPDN(dset, max_depth=max_depth)
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
                
            #print(iter)
            partitions = []
            for partition in range(0, len(numpy.unique(z))):
                # print(partition)
                # print(z == partition)
                partitions.append(data[z == partition, :])
                
            if len(partitions) < nM:
                return z
        
        return z    
    
