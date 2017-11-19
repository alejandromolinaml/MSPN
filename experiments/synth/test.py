'''
Created on May 9, 2017

@author: molina
'''

import os
import platform
if platform.system() == 'Darwin':
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
else:
    os.environ["R_HOME"] = "/usr/lib/R"
    
#print(os.environ["R_HOME"])

import numpy
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


import itertools
from joblib.memory import Memory
import math

from mlutils.transform import getOHE

path = os.path.dirname(__file__)
    
with open (path+"/rdcs.R", "r") as rfile:
    code = ''.join(rfile.readlines())
    rmodule = SignatureTranslatedAnonymousPackage(code, "rf")
    
numpy2ri.activate()

def rdccancor(x, y):
    #print(x,y)
    return rmodule.rdccancor(x, y)[0]

def rdcdcor(x, y):
    #print(x,y)
    return rmodule.rdcdcor(x, y)[0]
    

def generateGraphs(num_nodes):
    edges = list(itertools.combinations(numpy.arange(num_nodes), 2))
    nr_edges = len(edges)
    
    edgeDirections = list(itertools.product([0, 1, 2], repeat=nr_edges))
    
    edgeTypes = list(itertools.product([0, 1, 2], repeat=num_nodes))
    
    graphbase = numpy.zeros((nr_edges, 2))
    

    for i, p in enumerate(edges):
        graphbase[i, 0] = p[0]
        graphbase[i, 1] = p[1]
        
    result = []
    
    for i, t in enumerate(edgeTypes):
        for i, d in enumerate(edgeDirections):
            r = numpy.hstack((graphbase, numpy.zeros((graphbase.shape[0], 1))))
            r[:, 2] = numpy.asarray(d)
            result.append((t, r.astype(int)))
    
    return result
    
    


def generateSample(type, depvalues):
    
    if len(depvalues) == 0:
        if type == 0:
            # categorical
            return numpy.random.choice(5, 1)
        elif type == 1:
            # discrete
            return numpy.random.poisson(5, 1)
        elif type == 2:
            return numpy.random.normal(5, 10, 1)
        else:
            assert False, "invalid type"
    
    mix = numpy.dot(numpy.arange(len(depvalues)) + 1, depvalues)
    #print(mix)
    if type == 0:
        # categorical
        return numpy.random.normal(mix, 5, 1)
    
        catid = int(mix) % 5
        if catid == 0:
            return numpy.random.choice(5, 1, p=[0.1, 0.2, 0.3, 0.1, 0.3])
        if catid == 1:
            return numpy.random.choice(5, 1, p=[0.4, 0.1, 0.1, 0.1, 0.3])
        if catid == 2:
            return numpy.random.choice(5, 1, p=[0.8, 0.05, 0.05, 0.1, 0.0])
        if catid == 3:
            return numpy.random.choice(5, 1, p=[0.2, 0.3, 0.4, 0.05, 0.05])
        if catid == 4:
            return numpy.random.choice(5, 1, p=[0.3, 0.2, 0.1, 0.3, 0.1])
        if catid == 5:
            return numpy.random.choice(5, 1, p=[0.5, 0.0, 0.0, 0.2, 0.3])
        
        print(catid, mix)
        0/0
        
    
    elif type == 1:
        # discrete
        if mix == 0:
            mix = 1
            
        return numpy.random.poisson(abs(mix), 1)
    elif type == 2:
        # continuous
        return numpy.random.normal(mix, 10, 1)
    else:
        assert False, "invalid type"
    
def getDs(graph, samples=1000, burnin=100):
    numpy.random.seed(42)
    types = graph[0]
    adj = graph[1]
    num_nodes = len(types)
    
    # dependencies are encoding as:
    # N -> N 1
    # N <- N 2
    # N    N 0
    
    result = numpy.zeros(((samples + burnin)*num_nodes, num_nodes))
    
    
    ltredge = adj[:, 2] == 1
    rtledge = adj[:, 2] == 2
    
    deps = []
    
    for i in range(num_nodes):
        depltr = numpy.logical_and(adj[:, 1] == i, ltredge) 
        deprtl = numpy.logical_and(adj[:, 0] == i, rtledge) 
        deps.append(numpy.hstack((adj[depltr, 0], adj[deprtl, 1])))
    #print(deps)
    
    for r in range(1, (samples + burnin)):
        for c in range(num_nodes):
            ri = r*num_nodes+c
            result[ri, :] = result[ri-1, :]
            dsample = generateSample(types[c], result[ri - 1, deps[c]])
            result[ri, c] = dsample
            #print(dsample, result[ri, :])

    #print(result)
    result = result[burnin*num_nodes:, ]
    result = result[numpy.random.choice(result.shape[0], samples, replace=False),:]
    #print(result[:,])
    
    for i, typ in enumerate(types):
        if typ == 0:
            #print(result[:,i])
            h,b = numpy.histogram(result[:,i],5)
            result[:,i] = numpy.digitize(result[:,i], b)

    #0/0
    return result
     

def recoverIndp(graph, data, indtestfunc):
    numpy.random.seed(42)
    adj = graph[1]
    types = graph[0]
    
    edges = numpy.zeros((adj.shape[0]))
    
    edges[0] = 1
    
    threshold = [0, 1]
    for r in range(adj.shape[0]):
        
        c1 = adj[r, 0]
        c2 = adj[r, 1]
        
        d1 = data[:, c1]
        d2 = data[:, c2]
        
        if types[c1] == 0: #if categorical, do OHE
            d1 = getOHE(d1)
            
        if types[c2] == 0: #if categorical, do OHE
            d2 = getOHE(d2)
        
        #print(types)
        test = indtestfunc(d1, d2)
                
        edges[r] = test
        if adj[r, 2] == 0:
            threshold[0] = max(threshold[0], test)
        else:
            threshold[1] = min(threshold[1], test)
    
    rdcedges = numpy.copy(edges)
   # print(edges)
    
    maxmargt = (threshold[1] - threshold[0]) / 2 + threshold[0] 
        
    if threshold[1] < threshold[0]:
        maxmargt = (threshold[0] - threshold[1]) / 2 + threshold[1] 
    
    #print(threshold, maxmargt, edges)
    
    edges[edges <= maxmargt] = 0
    edges[edges > 0] = 1
    
    adjedges = numpy.copy(adj[:, 2])
    adjedges[adjedges > 0] = 1
    
    err = numpy.sum(numpy.abs(adjedges - edges))
    
    return (threshold, maxmargt, err, rdcedges, edges)




memory = Memory(cachedir="/tmp/spnht", verbose=0, compress=9)

@memory.cache
def getITstats(graphs, indtestfunc):

    results = numpy.zeros((len(graphs), 4))
    
    for i, g in enumerate(graphs):
        
        #print(getDs(g, samples=20, burnin=10))
        
        #i = 16
        #print(i)
        #g = graphs[i]
        #print(g)
        data = getDs(g, samples=2000, burnin=500)
        #0/0
        #print(data)
        
        dr = recoverIndp(g, data, indtestfunc)
        results[i, 0] = dr[0][0]
        results[i, 1] = dr[0][1]
        results[i, 2] = dr[1]
        results[i, 3] = dr[2]
        
        if dr[2] > 0:
            print(i, dr)
            print(g[0])
            print(g[1])
            print("--------------------------------------------------------------------------------------")
    
    return results



if __name__ == '__main__':
        
    
    graphs = generateGraphs(3)
    
    print(len(graphs))
    
    tstats = getITstats(graphs, rdccancor) 
    print(tstats)
    
    ix = tstats[:, 3] == 0
    
    print(numpy.average(tstats[ix, 0]), numpy.average(tstats[ix, 1]), numpy.average(tstats[ix, 2]))
    
    print(numpy.sum(tstats[:, 3] > 0))
    
    tstats = getITstats(graphs, rdcdcor) 
    
    
    ix = tstats[:, 3] == 0
    print(numpy.average(tstats[ix, 0]), numpy.average(tstats[ix, 1]), numpy.average(tstats[ix, 2]))
    
    
    print(numpy.sum(tstats[:, 3] > 0))
    
    
    
    
