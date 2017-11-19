'''
Created on 6 Jun 2017

@author: alejomc
'''
import itertools
import os
import platform
import sys
import warnings



if platform.system() == 'Darwin':
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
else:
    os.environ["R_HOME"] = "/usr/lib/R"



from matplotlib.backends.backend_pdf import PdfPages
import numpy

import matplotlib.pyplot as plt
from mlutils.datasets import loadMLC
import numpy as np




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #cm[cm>0]+=0.1
    if normalize:
        cm = cm / numpy.max(cm)
        #print(numpy.max(cm))
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar( pad=0.15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes[::-1], rotation=90)
    plt.yticks(tick_marks, classes)

    
    #print(cm)

    #thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, cm[i, j],
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')

((train, valid, test), feature_names, feature_types, domains) = loadMLC("autism", data_dir="datasets/autism/proc/unique")




adj = mem_map = numpy.memmap("mem.map", dtype='float', mode='r+', shape=(28, 28))

adj = numpy.copy(adj)

#adj = numpy.zeros((2,2))
#feature_names = ["0", "1"]
#adj[0,0] = 1
adj = numpy.fliplr(adj)

# Plot normalized confusion matrix
fig = plt.figure(figsize=(7,7))
plot_confusion_matrix(adj, classes=feature_names, normalize=True,
                      title='Normalized Mutual Information Autism Dataset')

pp = PdfPages('MI.pdf')
pp.savefig(fig)
pp.close()

