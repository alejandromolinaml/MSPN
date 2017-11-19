'''
Created on Jul 28, 2016

@author: molina

inspired by:
https://github.com/andrewclegg/sketchy/blob/master/sketchy.py

we just remove the seeds because inside the PSPN we want different projections at the different layers
'''

import math
import numpy
import random


def make_planes(N, dim):
    result = numpy.zeros((N, dim))
    for i in range(N):
        result[i, :] = numpy.random.uniform(-1, 1, dim)
    
    return result / numpy.sqrt(numpy.sum(result * result, axis=1))[:, None]
    


def above(planes, data):
    nD = data.shape[0]
    nP = planes.shape[0]
    centered = data - numpy.mean(data, axis=0)
    result = numpy.zeros((nD, nP))
    for i in range(nD):
        for j in range(nP):
            result[i, j] = numpy.sum(planes[j, :] * centered[i, :]) > 0
    return result


if __name__ == '__main__':
    planes = make_planes(1, 2)
    
    print(planes.shape)
    print(planes)
    
    data = numpy.random.random((20, 2))
    
    # print(data)
    
    isabove = above(planes, data)
    
    print(isabove[:, 0])





