'''
Created on Jan 17, 2017

@author: molina
'''
import numpy

from tfspn.SPN import SPN, Splitting
import tensorflow as tf


if __name__ == '__main__':

    gen = numpy.random.poisson(5, 1000)

    data = numpy.transpose(numpy.vstack((gen, gen)))
    

    spn = SPN.LearnStructure(data, min_instances_slice=200, families=["poisson", "poisson"], row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.IndependenceTest())
    
#    THIS PRODUCES THE FOLLOWING SPN:
#
#     SumNode_0 SumNode(0.154*ProductNode_3, 0.188*ProductNode_6, 0.158*ProductNode_10, 0.076*ProductNode_13, 0.13999999999999999*ProductNode_18, 0.176*ProductNode_21, 0.108*ProductNode_24){
#     ProductNode_3 ProductNode(PoissonNode_4, PoissonNode_5){
#         PoissonNode_4 P(X_0_|λ=6.0)
#         PoissonNode_5 P(X_1_|λ=6.0)
#         }
#     ProductNode_6 ProductNode(PoissonNode_7, PoissonNode_8){
#         PoissonNode_7 P(X_0_|λ=5.0)
#         PoissonNode_8 P(X_1_|λ=5.0)
#         }
#     ProductNode_10 ProductNode(PoissonNode_11, PoissonNode_12){
#         PoissonNode_11 P(X_0_|λ=7.360759493670886)
#         PoissonNode_12 P(X_1_|λ=7.360759493670886)
#         }
#     ProductNode_13 ProductNode(PoissonNode_14, PoissonNode_15){
#         PoissonNode_14 P(X_0_|λ=9.723684210526315)
#         PoissonNode_15 P(X_1_|λ=9.723684210526315)
#         }
#     ProductNode_18 ProductNode(PoissonNode_19, PoissonNode_20){
#         PoissonNode_19 P(X_0_|λ=3.0)
#         PoissonNode_20 P(X_1_|λ=3.0)
#         }
#     ProductNode_21 ProductNode(PoissonNode_22, PoissonNode_23){
#         PoissonNode_22 P(X_0_|λ=4.0)
#         PoissonNode_23 P(X_1_|λ=4.0)
#         }
#     ProductNode_24 ProductNode(PoissonNode_25, PoissonNode_26){
#         PoissonNode_25 P(X_0_|λ=1.6111111111111112)
#         PoissonNode_26 P(X_1_|λ=1.6111111111111112)
#         }
#     }
    
    
#this initializes the tensorflow graph    
    with tf.device("/cpu:0"):
        tf.reset_default_graph()
                    
        with tf.name_scope('input'):
            X = tf.placeholder(tf.float64, [None, 2], name="x")
        
        with tf.name_scope('SPN') as scope:
            spn.root.initTf(X)
            
       
    print(spn)
    
                
    with tf.Session() as sess:
        # variables need to be initialized before we can use them
        sess.run(tf.global_variables_initializer())
        
        #this computes the probabilities at the root
        probs_root =  spn.root.value.eval({X:data[0:5,]})
        print(spn.root)
        print(probs_root)
        
        probs_ProductNode_3 =  spn.root.children[0].value.eval({X:data[0:5,]})
        print(spn.root.children[0])
        print(probs_ProductNode_3)
        
        probs_ProductNode_6 =  spn.root.children[1].value.eval({X:data[0:5,]})
        print(spn.root.children[0])
        print(probs_ProductNode_6)
        
        
        
        
        
        
        
        