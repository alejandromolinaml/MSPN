'''
Created on Jan 12, 2017

@author: molina
'''
import time

import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics.classification import accuracy_score

from mlutils.benchmarks import Chrono
import tensorflow as tf
from tfspn.SPN import Splitting, SPN
from tfspn.tfspn import JointCost, GaussianNode, PoissonNode, BernoulliNode, \
    ProductNode, SumNode


if __name__ == '__main__':
    name = "twospirals"
    data = numpy.loadtxt("standard/"+name+".csv", delimiter=",")
    
    x = data[:,:2]
    y = data[:,2]
    
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=1337)
    
    traindata = numpy.c_[train_x, train_y]
    testdata = numpy.c_[test_x, test_y]
    #testdata = testdata[0:5,:]
    
    gn1 = GaussianNode("gn1", 0, "X0", 1.0, 1.0)
    pn1 = PoissonNode("pn1", 1, "X1", 1.0)
    bn1 = BernoulliNode("bn1", 2, "X2", 0.0)
    p1 = ProductNode("p1", gn1, pn1, bn1)
    
    gn2 = GaussianNode("gn2", 0, "X0", 10.0, 1.0)
    pn2 = PoissonNode("pn2", 1, "X1", 10.0)
    bn2 = BernoulliNode("bn2", 2, "X2", 1.0)
    p2 = ProductNode("p1", gn2, pn2, bn2)

    s1 = SumNode("s1", [0.5, 0.5], p1, p2)
    spn = SPN()
    spn.root = s1
    
    c = Chrono().start()
    
    with tf.device("/cpu:0"):
        tf.reset_default_graph()
                    
        with tf.name_scope('input'):
            X = tf.placeholder(tf.float64, [None, 3], name="x")
        
        with tf.name_scope('SPN') as scope:
            spn.root.initTf(X)
            costf = JointCost(spn.root)
        
        train_op = tf.train.AdamOptimizer().minimize(costf)
        
    print(c.end().elapsed())
    
    print(spn.root)
    
    with tf.Session() as sess:
        # variables need to be initialized before we can use them
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        
        nb_epochs = 10000+1
        
        for i in range(nb_epochs):
            c = Chrono().start()
            opt, cval  = sess.run([train_op, costf], feed_dict={X:traindata})
            #print(c.end().elapsed())
            #print(mn)
                
        print(spn.root)
        
        spn.root.tftopy()
    
        print(spn.root)
    
    
    
    tf.reset_default_graph()
                
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float64, [None, 3], name="x")
        
    with tf.name_scope('SPN') as scope:    
        spn.root.initMap(X, query=[2])
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # print(testdata)
        #print(spn.children[1].map.eval({X:testdata}))
        #print(spn.cmaps.eval({X:testdata}))
        #print(spn.maxprobs.eval({X:testdata}))
        #print(spn.root.map.eval({X:testdata}))
        #print(testdata)
        #print(spn.root.children[0])
        #print(spn.children[1])
        #print(spn.children[0].children[2].value.eval({X:testdata}))
        #print(spn.children[0].children[2].tfp.eval({X:testdata}))
        #print(spn.children[0].children[2].dist.p.eval({X:testdata}))
        #print(spn.children[0].children[2].dist.logits.eval({X:testdata}))
        #testdata[:,:] = 0
        #print(spn.children[0].value.eval({X:testdata}))
        
        
        
        predictions = spn.root.map.eval({X:testdata})[:, 2]
        
        print('MAP accuracy : ', accuracy_score(test_y, predictions))
        
        
    
        
        
        
