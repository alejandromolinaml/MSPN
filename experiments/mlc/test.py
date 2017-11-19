'''
Created on May 15, 2017

@author: molina
'''
import numpy

from mlutils.datasets import loadMLC
from tfspn.SPN import SPN, Splitting


dsname = "australian"
# dsname = "balance-scale"
# dsname = "breast"
# dsname = "cars"
# dsname = "cleve"
# dsname = "crx"
# dsname = "diabetes"
# dsname = "german-org"
# dsname = "glass"
# dsname = "glass2"
# dsname = "heart"
# dsname = "iris"

result = []
# , "balance-scale", "breast", "cars", "cleve", "crx", "diabetes", "german-org", "glass", "glass2", "heart", "iris"]:
for dsname in ["auto"]:

    (train, test, valid), feature_names, feature_types, domains = loadMLC(
        dsname, data_dir='datasets/MLC/proc-db/proc/auto/')

    #
    # train = train[:,(1,9)]
    # test = test[:,(1,9)]
    # valid = valid[:,(1,9)]
    # feature_names = [feature_names[1], feature_names[9]]
    # feature_types = [feature_types[1], feature_types[9]]
    # domains = [domains[1], domains[9]]
    #
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # plt.hist(train[:,0], bins=100, histtype='stepfilled', normed=True, color='b', label='Gaussian')
    # plt.hist(test[:,0], bins=100, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Uniform')
    #
    # plt.show()

    # print(domains)
    print(feature_names)
    print(feature_types)
    print(train.shape)

    # spn = SPN.LearnStructure(train, featureNames=feature_names, domains=domains,  featureTypes=feature_types, row_split_method=Splitting.RandomPartitionConditioningRows(), col_split_method=Splitting.RDCTestOHEpy(threshold=0.75),
    # spn = SPN.LearnStructure(train, featureNames=feature_names, domains=domains,  featureTypes=feature_types, row_split_method=Splitting.DBScanOHE(eps=1.0, min_samples=2), col_split_method=Splitting.RDCTestOHEpy(threshold=0.75),
    # spn = SPN.LearnStructure(train, featureNames=feature_names,
    # domains=domains,  featureTypes=feature_types,
    # row_split_method=Splitting.KmeansOHERows(),
    # col_split_method=Splitting.RDCTest(threshold=0.75),
    spn = SPN.LearnStructure(train, featureNames=feature_names, domains=domains,  featureTypes=feature_types, row_split_method=Splitting.Gower(), col_split_method=Splitting.RDCTest(threshold=0.05),
                             min_instances_slice=20,  cluster_first=True)

    print(spn)

    result.append([dsname, numpy.mean(spn.root.eval(train)), numpy.mean(
        spn.root.eval(valid)), numpy.mean(spn.root.eval(test))])
    print("train", numpy.mean(spn.root.eval(train)))
    print("valid", numpy.mean(spn.root.eval(valid)))
    print("test", numpy.mean(spn.root.eval(test)))

    print("train", numpy.min(spn.root.eval(train)))
    print("valid", numpy.min(spn.root.eval(valid)))
    print("test", numpy.min(spn.root.eval(test)))

print(result)
