'''
Created on Jun 28, 2017

@author: molina
'''
from joblib.memory import Memory
import numpy
import pickle
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition.online_lda import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from tfspn.SPN import SPN, Splitting
from tfspn.tfspn import ProductNode


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def test():
    n_topics = 10
    
    with open("acceptedoralpaperstext.txt") as f:
        content = []
        ids = []
        
        for line in f.readlines():
            cols = line.split("\t")
            content.append(cols[1].strip())
            ids.append(cols[0].strip())
        
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=100,
                                stop_words='english')
    
    
    #print(content)
    bow = tf_vectorizer.fit_transform(content)
    
    
    
    feature_names = tf_vectorizer.get_feature_names()
    
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=2000,
                                #learning_method='online',
                                #learning_offset=50.,
                                random_state=0)
    topics = lda.fit_transform(bow, bow)
    
    
    print(print_top_words(lda, feature_names, 10))
    print(topics)
    print(bow.shape)
    
    f = numpy.array(feature_names)
    
    data = numpy.array(bow.todense())
    
    featureTypes = ["discrete"] * data.shape[1]
    
    domains = []
    for i, ft in enumerate(featureTypes):
        domain = numpy.unique(data[:, i])

        # print(i, ft, domain)
        domains.append(domain)

        
    memory = Memory(cachedir="/tmp/spntopics", verbose=0, compress=9)


    @memory.cache
    def learn(data, min_instances_slice, feature_names, domains, featureTypes):
        spn = SPN.LearnStructure(data, featureTypes=featureTypes, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTest(threshold=0.1, linear=True),
                                featureNames=feature_names,
                                domains=domains,
                                 # spn = SPN.LearnStructure(data, featureNames=["X1"], domains =
                                 # domains, families=families, row_split_method=Splitting.KmeansRows(),
                                 # col_split_method=Splitting.RDCTest(),
                                 min_instances_slice=min_instances_slice)
        return spn
    
    
    
    
    print(data.shape)
    print(type(data))
    #0/0
    spn = learn(data, 5, f, domains, featureTypes)

    spn.root.validate()
    
    prodNodes = spn.get_nodes_by_type(ProductNode)

    for pn in prodNodes:
        leaves = pn.get_leaves()
        words = set()
        for leaf in leaves:
            # assuming pwl node:
            _x = numpy.argmax(leaf.y_range)
            max_x = leaf.x_range[_x]
            if max_x < 1.0:
                continue
            
            words.add(feature_names[leaf.featureIdx])
        # ll = pn.eval()
        if len(words) < 4:
            continue
        
        print(pn.rows, words)
        

    


test()

0/0 
distances = numpy.loadtxt("wmd-master/paperdistance.txt")
clusters  = linkage(distances)

plt.figure()
dn = dendrogram(clusters)
plt.show()
print(clusters)

0/0
