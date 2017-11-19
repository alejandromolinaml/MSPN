'''
Created on Jun 27, 2016

@author: molina
'''


import locale
locale.setlocale(locale.LC_NUMERIC, 'C')

from numba import jit
from math import exp, lgamma, log
from mlutils.statistics import poissonpmf
from mlutils.statistics import nppoissonpmf
from mlutils.statistics import gaussianpdf
from mlutils.statistics import loggaussianpdf
from mlutils.statistics import logbernoullipmf
from mlutils.statistics import bernoullipmf

EqCache = {}


def compileEq(eqstr, dic, compileC=True):
    import json
    import hashlib
        
    signature = "%s%s"%(eqstr, json.dumps(dic, sort_keys=True))
    
    if signature in EqCache:
        return EqCache[signature]
        
    fid = hashlib.sha224(signature.encode('utf-8')).hexdigest()
        
    for key, value in dic.items():
        eqstr = eqstr.replace(key, value)
        
    code = "def compiledEq_%s(%s): return %s" % (fid, ",".join(dic.values()), eqstr)
    
    exec(code)
        
    func = eval("compiledEq_%s" % (fid))

    if compileC:
        func = jit(func, nopython=True)
    
    EqCache[signature] = func
        
    return func


def compileEqNumpy(eqstr, compileC=True):
    import hashlib
        
    signature = eqstr
    
    if signature in EqCache:
        return EqCache[signature]
        
    fid = hashlib.sha224(signature.encode('utf-8')).hexdigest()
        
    code = "def compiledEq_%s(x): return %s" % (fid, eqstr)
    exec(code)
        
    func = eval("compiledEq_%s" % (fid))

    if compileC:
        func = jit(func, nopython=True)
    
    EqCache[signature] = func
        
    return func
