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
    
# print(os.environ["R_HOME"])

import numpy
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

path = os.path.dirname(__file__)
    
with open (path + "/histogram.R", "r") as rfile:
    code = ''.join(rfile.readlines())
    rmodule = SignatureTranslatedAnonymousPackage(code, "rf")
    
numpy2ri.activate()



def getHistogramVals(data):
    result = rmodule.getHistogram(data)
    breaks = numpy.asarray(result[0])
    densities = numpy.asarray(result[2])
    mids = numpy.asarray(result[3])
    
    return breaks, densities, mids

if __name__ == '__main__':
        
    dat = numpy.loadtxt("/Users/alejomc/git/TF_SPN/TFSPN/src/experiments/dataPWL.txt")
    print(dat)
    breaks, densities, mids = getHistogramVals(dat)
    
    print(breaks, densities, mids)
    
    xmax = 1.0
    xmin = 0.0
    areaOutside = (xmax-xmin)
    print(areaOutside)
    for i, d in enumerate(densities):
        if d == 0.0:
            bucketArea = breaks[i+1]-breaks[i]
            areaOutside -= bucketArea
            print(i,d, bucketArea)
    densityOutside = 1.0/areaOutside
    print(areaOutside, densityOutside)
    
    import bisect
    
    
    densityOutside = 1.0 / areaOutside
    w = 0.00000000
    out = 0
    ll = numpy.ones_like(dat)*(numpy.log(w*densityOutside))
    for i, x in enumerate(dat):
        outsideLeft = bisect.bisect(breaks, x) == 0
        outsideRight = bisect.bisect_left(breaks, x) == len(breaks)
        outside = outsideLeft or outsideRight
        density = densities[bisect.bisect_left(breaks,x)-1]
        
        if outside or density == 0.0:
            continue
    
        print(density)
        ll[i] = numpy.log((1.0-w)*density)
        #print(x, density)
    
    print(numpy.mean(ll))
    
    print(dat.shape, out)




