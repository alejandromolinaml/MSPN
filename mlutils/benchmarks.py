'''
Created on 13.06.2016

@author: alejomc
'''
from collections import OrderedDict
import json
import time


class Chrono(object):
    def __init__(self):
        self.started = 0
        self.finished = 0
        
    def start(self):
        self.started = time.time()
        return self
    
    def end(self):
        self.finished = time.time()
        return self
    
    #returns time in seconds
    def elapsed(self):
        return self.finished - self.started

class Stats(object):
    LOG_LIKELIHOOD = "Log Likelihood"
    MEAN_LOG_LIKELIHOOD = "Mean Log Likelihood"
    TIME = "Time"
    MODEL_SIZE = "Model Size"
    PERPLEXITY = "Perplexity"
    ABS_ERROR = "$|x-\mu|$"
    SQUARED_ERROR = "$(x-\mu)^2$"
    F1_SCORE = "F1Score"
    ACCURACY = "Accuracy"
    PRECISION = "Precision"
    RECALL = "Recall"


    
    def __init__(self, name=None, fname=None):
        self.stats = OrderedDict()
        self.config = OrderedDict()
        self.name = name
        
        if fname is not None:
            self.load(fname)
        
    def addConfig(self, method, config):
        self.config[method] = config
    
    def add(self, method, measure, value):
        
        if measure not in self.stats:
            self.stats[measure] = OrderedDict()
            
        if method not in self.stats[measure]:
            self.stats[measure][method] = []
            
        self.stats[measure][method].append(value)
        
        return (method,measure,value)
        
    def get(self, method, measure):
        return self.stats[measure][method]
    
    def getMeasures(self):
        for measure in self.stats.keys():
            yield measure
    
    def getMethods(self, measure):
        for method in self.stats[measure]:
            yield method
            
    def getValues(self, method, measure):
        return self.stats[measure][method]
    
    def save(self, file):
        with open(file, 'w') as outfile:
            json.dump([self.name, self.stats, self.config], outfile)
            
    def load(self, file):
        with open(file) as data_file:    
            self.name, self.stats, self.config = json.load(data_file)
        
