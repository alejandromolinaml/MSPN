# PSPN

Python implementation of the  Poisson Sum-Product Networks (PSPN). It provides
routines to do inference and learning.

## Overview

**Paper reference here**


## Requirements
[numpy](http://www.numpy.org/),
[sklearn](http://scikit-learn.org/stable/),
[scipy](http://www.scipy.org/), 
[numba](http://numba.pydata.org/), 
[matplotlib](http://matplotlib.org/)
[joblib](https://pythonhosted.org/joblib/)
[networkx](https://networkx.github.io/)
[pandas](pandas.pydata.org)
[sympy](www.sympy.org)
[statsmodels](http://statsmodels.sourceforge.net)
[h2o](http://www.h2o.ai/) Required in some experiments
[gensim](https://radimrehurek.com/gensim/) Required in some experiments
[mpmath](http://mpmath.org)

extra packages used for plotting the graph structure:
 pydotplus
 PyPDF2
 reportlab
 graphviz




## Usage
please look in the experiments folder, there you will find code and data for the experiments in the paper.
usually there are two files, one for computing the results and one for plotting the results

to Learn, do:
from algo.learnspn import LearnSPN

spn = LearnSPN(alpha=0.001, min_instances_slice=50).fit_structure(train_data)

to Learn using cache, do:
from algo.learnspn import LearnSPN

memory = Memory(cachedir="/tmp", verbose=0, compress=9)

spn = LearnSPN(alpha=0.001, min_instances_slice=50, cache=memory).fit_structure(data)

to do inference, do:
spn.eval(test_data)


Good starting points:
/spyn/experiments/MI/nipsGraph.py
/spyn/experiments/dependencytypes/plotDifferentDistributionTypes.py

more details in:

- The learning algorithm is in: algo/learnspn.py the learning method is called "fit_structure"
- Inference algorithms are located in: spn/linked/spn.py:
	to compute log likelihood use the method "eval" to get LL per data row use individual=True, to get LL for the whole dataset use individual=False
	to compute perplexity use the method "perplexity"
	to do Max Prod use the method "complete" and pass instances with "None" where you want to get the MPE
	
	to compute Mutual Information I(X,Y) = Sum_x,y Pxy * (log2(Pxy) - (log2(Px) + log2(Py))) use "computeMI" passing the feature names you want to compute and the array of feature names
	to compute Entropy H(X) = Sum_x Px * log2(Px) use "computeEntropy" passing the feature name you want to compute and the array of feature names
	to compute Entropy H(X, Y) = Sum_x,y Pxy * log2(Pxy) use "computeEntropy2" passing the feature names you want to compute and the array of feature names
	to compute Expectation E(X) = Sum_x x * Px use "computeExpectation" passing the feature name you want to compute and the array of feature names
	to compute Expectation E(X, Y) = Sum_x,y (x * y) * Pxy use "computeExpectation2" passing the feature names you want to compute and the array of feature names
	to compute Covariance Cov(X, Y) = E(X,Y) - E(X) * E(Y) use "computeCov" passing the feature names you want to compute and the array of feature names
	to compute distance d(x,y) = H(x) + H(y) - 2.0 * I(x,y) use "computeDistance" passing the feature names you want to compute and the array of feature names
	to compute the normalized distance D(x,y) = d(x,y) / H(x,y) use "computeNormalizedDistance" passing the feature names you want to compute and the array of feature names
		
	to get the number of node in the spn use "size", this gives an idea of how big/deep the model is
	to get the SPN in a text representation use "to_text" and pass the names of the features as an array of strings
	to get the SPN in a graph representation use "to_graph" this returns a networkx DiGraph and you can use "save_pdf_graph" to plot the graph
	





	 

