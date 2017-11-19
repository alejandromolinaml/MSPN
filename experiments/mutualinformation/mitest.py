'''
Created on 5 Jun 2017

@author: alejomc
'''
import os
import platform
import warnings
import sys


if platform.system() == 'Darwin':
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
else:
    os.environ["R_HOME"] = "/usr/lib/R"
    

from joblib.memory import Memory
import numpy
from rpy2.rinterface import RRuntimeWarning

from mlutils.datasets import loadMLC
from tfspn.SPN import SPN, Splitting
from tfspn.measurements import computeMI


numpy.set_printoptions(precision=10, suppress=True)



    
warnings.filterwarnings("ignore", category=RRuntimeWarning)

((train, valid, test), feature_names, feature_types, domains) = loadMLC("autism", data_dir="datasets/autism/proc/unique")

nfeatures = len(feature_types)
print(nfeatures)

#0/0

#memory = Memory(cachedir="/data/ssd/molina/tmp/mi", verbose=0, compress=9)
memory = Memory(cachedir="/tmp/mi", verbose=0, compress=9)

data = numpy.vstack((train, valid, test))



domains = [numpy.unique(data[:,i]) for i in range(data.shape[1])]

#print("dataminmax", numpy.min(data[:,16]), numpy.max(data[:,16]))

#print(domains[0])
#print(domains[16])

@memory.cache
def learn(data, featureTypes, families, domains, feature_names, min_instances_slice, prior_weight=0.0):
    return SPN.LearnStructure(data, prior_weight=prior_weight, featureTypes=featureTypes, row_split_method=Splitting.KmeansRDCRows(), col_split_method=Splitting.RDCTest(threshold=0.3),
                             domains=domains,
                             families=families,
                             featureNames=feature_names,
                             min_instances_slice=min_instances_slice)

#print("learning")
spn = learn(data, featureTypes=feature_types, families=["piecewise"]*train.shape[1], domains=domains, feature_names=feature_names, min_instances_slice=100)

print("done learning")
i = 1
j = 4
computeMI(spn, i, j, verbose=True)
0/0

mem_map = numpy.memmap(sys.argv[1], dtype='float', mode='r+', shape=(nfeatures, nfeatures))

i = int(sys.argv[2])
j = int(sys.argv[3])

mem_map[i,j] = computeMI(spn, i, j, verbose=True)
mem_map.flush()

#print(spn)

#/Users/alejomc/git/TF_SPN/TFSPN/src/mlutils/datasets/autism/autism.features