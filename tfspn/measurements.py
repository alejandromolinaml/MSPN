from math import sqrt

from mpmath import nsum, inf
from numba import jit
import numpy
from scipy import integrate

from mlutils.fastmath import compileEq


def getJointDist(self, f1, f2):
    Pxy = self.marginalizeToEquation([f1, f2])
    
    func = compileEq(Pxy, {"x_%s_" % f1: "x", "x_%s_" % f2: "y"})
            
    return func
    
    
def computeEntropy(spn, f1, verbose=False):
        
    Px = spn.marginalizeToEquation([f1])
    
    sumHxy = "{Px} * log({Px})/log(2)".format(**{'Px': Px})
    
    evl = compileEq(sumHxy, {"x_%s_" % f1: "x"})
    
    Hx = -nsum(lambda x: evl(int(x)), [0, inf], verbose=False, method="d", tol=10 ** (-10))
    
    if verbose:
        print("H(%s)=%s" % (f1, Hx))

    return Hx

def computeEntropy2(spn, f1, f2, verbose=False):
    
    Pxy = spn.marginalizeToEquation([f1, f2])
    
    sumHxy = "{Pxy} * log({Pxy})/log(2)".format(**{'Pxy': Pxy})
    
    evl = compileEq(sumHxy, {"x_%s_" % f1: "x", "x_%s_" % f2: "y"})
    # evl = lambda x, y: eval(sumHxy, None, {"x_%s_" % f1:x, "x_%s_" % f2: y, "poissonpmf": poissonpmf})
        
    Hxy = -nsum(lambda x, y: evl(int(x), int(y)), [0, inf], [0, inf], verbose=False, methomethod="d", tol=10 ** (-10))    
    if verbose:
        print("H(%s,%s)=%s" % (f1, f2, Hxy))

    return Hxy

inner = 0
outer = 0

def computeMI(spn, f1, f2, verbose=False):
    global inner
    global outer

    pxy = spn.marginalize([f1, f2])
    px = spn.marginalize([f1])
    py = spn.marginalize([f2])
    
    inp = numpy.zeros((1, len(spn.config["families"]) ))
    
    @jit
    def evspn(f1v, f2v):
        l2 = numpy.log(2)
        
        inp[:] = 0
        inp[0,f1] = f1v
        logpxval = px.eval(inp)[0]
        
        inp[:] = 0
        inp[0,f2] = f2v
        logpyval = py.eval(inp)[0]
        
        inp[:] = 0
        inp[0,f1] = f1v
        inp[0,f2] = f2v
        logpxyval = pxy.eval(inp)[0]
        pxyval = numpy.exp(logpxyval)
        
        #return pxyval
        return pxyval * (logpxyval/l2 - (logpxval/l2 + logpyval/l2))
    
    
    f1domains = spn.config["domains"][f1]
    f2domains = spn.config["domains"][f2]

    f1minv = int(numpy.min(f1domains))
    f1maxv = int(numpy.max(f1domains))+1
    f1range = range(f1minv, f1maxv)

    f2minv = int(numpy.min(f2domains))
    f2maxv = int(numpy.max(f2domains))+1
    f2range = range(f2minv, f2maxv)
    
    
    def evalAllF2(x):

        if spn.config["families"][f2] == "poisson":
            return nsum(lambda f2val: evspn(x,f2val), [0, inf], verbose=False, method="d")

        if spn.config["families"][f2] == "isotonic":
            return sum(map(lambda f2val: evspn(x, f2val), f2range))

        global inner
        if spn.config["featureTypes"][f2] == "continuous":
            minv = numpy.min(f2domains)
            maxv = numpy.max(f2domains)
            integral = integrate.quad(lambda f2val: evspn(x, f2val), minv, maxv, limit=30, maxp1=30, limlst=30)
            print(outer, inner, x, integral)
            inner += 1
            return integral[0]
            #return nsum(lambda f2val: evspn(x,f2val), [minv, maxv], verbose=False, method="d", tol=10 ** (-10))
        else:
            return sum(map(lambda f2val: evspn(x, f2val), f2domains))
    
    outer = 0
    def computeIxy():
        if spn.config["families"][f1] == "poisson":
            return nsum(lambda f1val: evalAllF2(f1val), [0, inf], verbose=False, method="d")

        if spn.config["families"][f1] == "isotonic":
            #print(list(range(minv, maxv)))
            return sum(map(lambda f1val: evalAllF2(f1val), f1range))


        global outer
        global inner
        if spn.config["featureTypes"][f1] == "continuous":
            minv = numpy.min(f1domains)
            maxv = numpy.max(f1domains)
            inner = 0
            integral = integrate.quad(lambda f1val: evalAllF2(f1val), minv, maxv, limit=30, maxp1=30, limlst=30)
            print(outer, inner, integral)
            outer += 1
            return integral[0]
            #return nsum(lambda f1val: evalAllF2(f1val), [minv, maxv], verbose=False, method="d", tol=10 ** (-10))
        else:
            return sum(map(lambda f1val: evalAllF2(f1val), f1domains))
    
    Ixy = computeIxy()
    
    if verbose:
        print("I(%s,%s)=%s" % (f1, f2, Ixy))
        
    return Ixy

def computeMI2(spn, f1, f2, verbose=False):
    
   
    Pxy = spn.marginalizeToEquation([f1, f2])
    Px = spn.marginalizeToEquation([f1])
    Py = spn.marginalizeToEquation([f2])

    sumIxy = "{Pxy} * (log({Pxy})/log(2) - (log({Px})/log(2) + log({Py})/log(2)))".format(**{'Pxy': Pxy, 'Px': Px, 'Py': Py})
    # evl = lambda x, y: eval(sumIxy, None, {"x_%s_" % f1:x, "x_%s_" % f2: y, "poissonpmf":poissonpmf})
    evl = compileEq(sumIxy, {"x_%s_" % f1: "x", "x_%s_" % f2: "y"}, compileC=False)
    
    
    Ixy = nsum(lambda x, y: evl(int(x), int(y)), [0, inf], [0, inf], verbose=False, method="d", tol=10 ** (-10))
    
    if verbose:
        print("I(%s,%s)=%s" % (f1, f2, Ixy))
        
    return Ixy

def computeExpectation(spn, f1, verbose=False):
    
    Px = spn.marginalizeToEquation([f1])
    
    sumEx = "x * {Px}".format(**{'Px': Px})

    evl = compileEq(sumEx, {"x_%s_" % f1: "x"})
    
    Ex = nsum(lambda x: evl(int(x)), [0, inf], verbose=False, method="d", tol=10 ** (-10))

    if verbose:
        print("E(%s)=%s" % (f1, Ex))

    return Ex

def computeExpectation2(spn, f1, f2, verbose=False):
    
    Pxy = spn.marginalizeToEquation([f1, f2]).replace("x_%s_" % f1, "x").replace("x_%s_" % f2, "y")
    
    sumExy = "(x * y) * {Pxy}".format(**{'Pxy': Pxy})
    
    evl = compileEq(sumExy, {"x_%s_" % f1: "x", "x_%s_" % f2: "y"})
    
    Exy = nsum(lambda x, y: evl(int(x), int(y)), [0, inf], [0, inf], verbose=False, method="d", tol=10 ** (-10))

    if verbose:
        print("E(%s, %s)=%s" % (f1, f2, Exy))

    return Exy

def computeCov(spn, f1, f2, verbose=False):
    EXY = computeExpectation2(spn, f1, f2, verbose)
    EX = computeExpectation(spn, f1, verbose)
    EY = computeExpectation(spn, f2, verbose)
     
    cov = EXY - EX * EY
    if verbose:
        print("Cov(%s, %s)=%s" % (f1, f2, cov))
        
    return cov


def computeNMI(spn, fn1, fn2, verbose=False):
        
    Ixy = computeMI(spn, fn1, fn2, verbose)
    Hx = computeEntropy(spn, fn1, verbose)
    Hy = computeEntropy(spn, fn2, verbose)
    
    NMIxy = Ixy / sqrt(Hx * Hy)

    if verbose:
        print("NMI(%s,%s)=%s" % (fn1, fn2, NMIxy))

    return NMIxy


def computeDistance(spn, fn1, fn2, verbose=False):
    
    Ixy = computeMI(spn, fn1, fn2, verbose)
    Hx = computeEntropy(spn, fn1, verbose)
    Hy = computeEntropy(spn, fn2, verbose)
    
    Dxy = Hx + Hy - 2.0 * Ixy

    if verbose:
        print("d(%s,%s)=%s" % (fn1, fn2, Dxy))

    return Dxy

def computeNormalizedDistance(spn, fn1, fn2, verbose=False):
    # http://montana.informatics.indiana.edu/LabWebPage/Presentations/Vikas_Nov02_2011.pdf
    dxy = computeDistance(spn, fn1, fn2, verbose)
    Hxy = computeEntropy2(spn, fn1, fn2, verbose)
    
    Dxy = dxy / Hxy

    if verbose:
        print("D(%s,%s)=%s" % (fn1, fn2, Dxy))

    return Dxy



