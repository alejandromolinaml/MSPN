import numpy
from mlutils.datasets import getCIFAR10
from tfspn.SPN import SPN, Splitting


dsname, train, test, labels_train, labels_test = getCIFAR10(grayscale=True)

data = numpy.vstack((train, test))


ds = numpy.hstack((train, labels_train))

domains = [numpy.unique(ds[:, i]) for i in range(ds.shape[1])]


spn = SPN.LearnStructure(ds, prior_weight=0.0, featureTypes=["gaussian"] * train.shape[1]+["discrete"],
                              row_split_method=Splitting.RandomPartitionRows(), col_split_method=Splitting.RDCTest(threshold=0.3, OHE=True),
                              domains=domains,
                              families=["gaussian"] * ds.shape[1],
                              min_instances_slice=5000000)


print("learned")

ts = numpy.hstack(test, numpy.zeros_like(labels_test)/0)

ts = ts[0:10,:]

print(ts[0,:])

predicted_labels = spn.root.mpe_eval(ts)

print(predicted_labels[0,:])
