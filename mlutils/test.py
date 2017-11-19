'''
Created on 12.06.2016

@author: alejomc
'''
from numpy import asarray
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import KFold


def kfolded(data, folds, seed=1337):
    kf = KFold(n_splits=folds, random_state=seed)
    
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        if len(data.shape) > 1:
            yield (data[train_index,:], data[test_index,:], i)
        else:
            yield (data[train_index], data[test_index], i)
            
def classkfolded(X, Y, folds, seed=1337):
    kf = StratifiedKFold(n_splits=folds)
    
    for i, (train_index, test_index) in enumerate(kf.split(X, Y)):
        if len(X.shape) > 1:
            yield (X[train_index,:], Y[train_index], X[test_index,:], Y[test_index], i)
        else:
            yield (X[train_index], Y[train_index], X[test_index], Y[test_index], i)


if __name__ == '__main__':

    for train, test, i in kfolded(asarray(range(10)), 4):
        print(train, test, i)