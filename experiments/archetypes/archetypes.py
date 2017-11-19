'''
Created on May 23, 2017

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


import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)


path = os.path.dirname(__file__)
    
with open (path+"/archetypes.R", "r") as rfile:
    code = ''.join(rfile.readlines())
    rmodule = SignatureTranslatedAnonymousPackage(code, "rf")
    
numpy2ri.activate()



def getArchetypes(data, num, seed=1):
    try:
        if seed > 100:
            return None, None
        
        #print(seed)
        result = rmodule.getArchetypes(data, num)
        arcs = numpy.asarray(result[0])
        mixt = numpy.asarray(result[1])
        return (arcs, mixt)
    except:
        return getArchetypes(data, num, seed+1)
    
def getDirichlet(data, seed=1):
    if seed > 100:
        return None, None
    
    result = rmodule.getDirichlet(data, seed)
    return numpy.asarray(result)

