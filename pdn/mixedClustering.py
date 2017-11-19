import os
import platform
from mlutils import datasets

if platform.system() == 'Darwin':
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
    #os.environ["R_HOME"] = "/usr/local/Cellar/r/3.3.1_2/R.framework/Resources"

#else:
#    os.environ["R_HOME"] = "/usr/lib/R"
    
#print(os.environ["R_HOME"])

import numpy
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


path = os.path.dirname(__file__)

with open (path+"/mixedClustering.R", "r") as mixfile:
    code = ''.join(mixfile.readlines())
    mixedClustring = SignatureTranslatedAnonymousPackage(code, "mixedClustring")



def getMixedGowerClustering(data,featureTypes, n_clusters=2, random_state=42):
    
    
    assert data.shape[1] == len(featureTypes), "invalid parameters"
    
    numpy2ri.activate()
    try:
        df = robjects.r["as.data.frame"](data)
        #print(featureTypes, n_clusters, random_state)
        result = mixedClustring.mixedclustering(df, featureTypes, n_clusters, random_state)
        result = numpy.asarray(result)
    except Exception as e:
        numpy.savetxt("/tmp/errordata.txt", data)
        print(e)
        raise e

    return result


if __name__ == '__main__':
    
    data = numpy.loadtxt("/tmp/errordata.txt")
    print(data.shape)
    
    
    featuretypes = ['categorical', 'continuous', 'continuous', 'categorical', 'categorical', 'categorical', 'continuous', 'categorical', 'categorical', 'continuous', 'categorical', 'categorical', 'continuous', 'continuous', 'categorical']
    
    
    print(len(featuretypes))
    
    getMixedGowerClustering(data, featuretypes)

    
    0/0
    for dsname, data, featureNames, classes, families in [getIRIS(2)]:
        families = ["categorical", "continuous", "continuous", "continuous", "continuous"]
        print(data.shape, len(families), families)
        out = getMixedGowerClustering(data, families, 2, 42)
        print(out)
