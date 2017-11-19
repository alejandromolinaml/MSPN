'''
Created on Jan 12, 2017

@author: molina
'''
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics.classification import accuracy_score
import time

from mlutils.benchmarks import Chrono
import tensorflow as tf
from tfspn.SPN import Splitting, SPN
from tfspn.tfspn import JointCost, DiscriminativeCost


def predict(spn, testdata, featureId):
    tf.reset_default_graph()
                
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float64, [None, testdata.shape[1]], name="x")
        
    with tf.name_scope('SPN') as scope:    
        spn.root.initMap(X, query=[featureId])
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        predictions = spn.root.map.eval({X:testdata})[:, featureId]
        
        return predictions
        
        

if __name__ == '__main__':
    name = "twospirals"
    data = numpy.loadtxt("standard/"+name+".csv", delimiter=",")
    
    x = data[:,:2]
    y = data[:,2]
    
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=1337)
    
    traindata = numpy.c_[train_x, train_y]
    testdata = numpy.c_[test_x, test_y]
    
    #print(traindata)
    #print(testdata)
    
    spn = SPN.LearnStructure(traindata, min_instances_slice=200, families=["gaussian", "gaussian", "bernoulli"], row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.IndependenceTest())
    
    print(spn)
    
    predictions = predict(spn, testdata, 2)
    print('MAP accuracy : ', accuracy_score(test_y, predictions))
    
    
    tf.reset_default_graph()
    
    c = Chrono().start()
    
    with tf.device("/cpu:0"):
        tf.reset_default_graph()
                    
        with tf.name_scope('input'):
            X = tf.placeholder(tf.float64, [None, 3], name="x")
        
        with tf.name_scope('SPN') as scope:
            spn.root.initTFSharedData(X)
            spn.root.initTf()
            costf = JointCost(spn.root)
            
            print("marginal")    
            spnMarginal = spn.root.marginalizeOut([2])
            print(spnMarginal)
            spnMarginal.initTFSharedData(X)
            spnMarginal.initTf()
    
            costf = DiscriminativeCost(spn.root, spnMarginal)
            
            #costf = -tf.reduce_sum(tf.log(spn.root.children[0].value))
        
        train_op = tf.train.AdamOptimizer().minimize(costf)
        
    print(c.end().elapsed())
        
            
    with tf.Session() as sess:
        # variables need to be initialized before we can use them
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        
        nb_epochs = 1000+1
        
        for i in range(nb_epochs):
            c = Chrono().start()
            opt, cval  = sess.run([train_op, costf], feed_dict={X:traindata})
            #print(c.end().elapsed())
            #print(mn)
                
        spn.root.tftopy()
    
    print(spn)
    
    tf.reset_default_graph()
    
    predictions = predict(spn, testdata, 2)
    print('MAP accuracy : ', accuracy_score(test_y, predictions))
        
        
    
        
        
        