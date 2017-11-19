'''
Created on Apr 28, 2017

@author: molina
'''
import numpy

from pdn.independenceptest import getIndependentRDCGroups


def test1():
    numpy.random.seed(42)
    instances = 2000
    data = numpy.hstack((numpy.random.normal(0, 1, instances).reshape(instances,1), numpy.random.normal(10, 1, instances).reshape(instances,1)))
    
    clusters = getIndependentRDCGroups(data, 0.3)
    print(numpy.loadtxt("/tmp/adj.txt", skiprows=1, delimiter=','))
    print()
    
    data = numpy.hstack((data, numpy.ones((instances, 1))))
    print(data.shape)
    
    data[0,2] = 0
    clusters = getIndependentRDCGroups(data, 0.3)
    print(numpy.loadtxt("/tmp/adj.txt", skiprows=1, delimiter=','))
    print()
    
    d2 = data.copy() 
    d2[1:100,2] = 0
    clusters = getIndependentRDCGroups(d2, 0.3)
    print(numpy.loadtxt("/tmp/adj.txt", skiprows=1, delimiter=','))
    print()

    d3 = numpy.hstack((data, 2*data[:,0].reshape(instances,1)))
    clusters = getIndependentRDCGroups(d3, 0.3)
    print(numpy.loadtxt("/tmp/adj.txt", skiprows=1, delimiter=','))
    print()
    
test1()