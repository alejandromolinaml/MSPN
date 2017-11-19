'''
Created on Feb 15, 2016

@author: molina
'''
import math
import numpy
from scipy.stats._discrete_distns import poisson
import statsmodels.api as sm


class GLMPDN:
    
    def __init__(self, data):
        self.data = data
        self.nD = data.shape[0]
        self.nF = data.shape[1]
        self.initial_model = numpy.log(numpy.mean(data, axis=0))
        self.glms = []
        
        self.train()
    
    

    def train(self):
        self.glms = []
        mask = numpy.array([True] * self.nF)
        for i in range(0, self.nF):
            mask[:] = True
            mask[i] = False
            glm = sm.GLM(self.data[:, i], self.data[:, mask], family=sm.families.Poisson(link=sm.families.links.log)).fit()
            self.glms.append(glm)
    
    def getLambdas(self, document):
        result = numpy.zeros(self.nF)
        
        mask = numpy.array([True] * self.nF)
        for i in range(0, self.nF):
            mask[:] = True
            mask[i] = False
            result[i] = self.glms[i].predict(document[mask])

        return result
    
    
    def getLogLikelihood(self, documents, debug=False):
        ll = 0
        
        if documents.ndim > 1:
            for i in range(0, documents.shape[0]):
                ll += self.getLogLikelihood(documents[i])
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
                    pmfval = numpy.finfo(float).eps
                ll += math.log(pmfval)
        return ll

     
    
