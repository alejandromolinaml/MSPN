'''
Created on May 11, 2017

@author: molina
'''
import numpy


def getOHE(data, domain=None):
    domain = numpy.unique(data)

    dataenc = numpy.zeros((data.shape[0], len(domain)))

    dataenc[data[:, None] == domain[None, :]] = 1

    assert numpy.all((numpy.sum(dataenc, axis=1) == 1)), "one hot encoding bug"

    return dataenc