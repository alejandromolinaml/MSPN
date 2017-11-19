import os
import platform
from multiprocessing.pool import Pool
from mlutils.datasets import getNips

from sklearn.model_selection import train_test_split

if platform.system() == 'Darwin':
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
else:
    os.environ["R_HOME"] = "/usr/lib/R"

from joblib.memory import Memory
import numpy

from tfspn.SPN import SPN, Splitting
from tfspn.measurements import computeMI

memory = Memory(cachedir="/tmp/mi", verbose=0, compress=9)





ds, data, words = getNips()

domains = [numpy.unique(data[:, i]) for i in range(data.shape[1])]

#print(domains)

train, test = train_test_split(data, test_size=0.2, random_state=42)



# print("dataminmax", numpy.min(data[:,16]), numpy.max(data[:,16]))

# print(domains[0])
# print(domains[16])

@memory.cache
def learn(data, featureTypes, families, domains, feature_names, min_instances_slice, row_split_method, col_split_method,
          prior_weight=0.0):
    return SPN.LearnStructure(data, prior_weight=prior_weight, featureTypes=featureTypes,
                              row_split_method=row_split_method, col_split_method=col_split_method,
                              domains=domains,
                              families=families,
                              featureNames=feature_names,
                              min_instances_slice=min_instances_slice)


# print("learning")
pspn = learn(train, featureTypes=["discrete"] * data.shape[1], families=["poisson"] * data.shape[1], domains=domains,
             feature_names=words, min_instances_slice=200, row_split_method=Splitting.KmeansRows(),
             col_split_method=Splitting.IndependenceTest(0.001))



marg = pspn.marginalize([0,1,2,3])

print(marg.toEquation())
print(marg)

0/0

mspn = learn(train, featureTypes=["discrete"] * data.shape[1], families=["isotonic"] * data.shape[1], domains=domains,
             feature_names=words, min_instances_slice=200, row_split_method=Splitting.KmeansRDCRows(),
             col_split_method=Splitting.RDCTest(threshold=0.1, OHE=False))



#print(pspn)
# print(mspn)

print("sum LL pspn", numpy.sum(pspn.root.eval(test)) )
print("sum LL mspn", numpy.sum(mspn.root.eval(test)) )
print("mean LL pspn", numpy.mean(pspn.root.eval(test)) )
print("mean LL mspn", numpy.mean(mspn.root.eval(test)) )

0/0

def getmiforfeature(input):
    spn, i, j = input
    # return i+j
    return computeMI(spn, i, j, verbose=True)


@memory.cache
def getmi(spn, numFeatures):
    adj = numpy.zeros((numFeatures, numFeatures))
    tasks = []
    for i in range(numFeatures):
        print(i)
        for j in range(i + 1, numFeatures):
            tasks.append((spn, i, j))
    print(len(tasks), "tasks reading")
    p = Pool()
    mapping = p.map(getmiforfeature, tasks)

    for ti, task in enumerate(tasks):
        _, i, j = task
        adj[i, j] = adj[j, i] = mapping[ti]
        # adj[j,i] = 1

    return adj

for n in pspn.top_down_nodes():
    n.__set_serializable__()

for n in mspn.top_down_nodes():
    n.__set_serializable__()


pspnmi = getmi(pspn, len(words))
mspnmi = getmi(mspn, len(words))



print("Frobenius norm", numpy.linalg.norm(pspnmi - mspnmi))

# print(pspnmi)
# computeMI(pspn, 0, 2, verbose=True)
# computeMI(mspn, 0, 2, verbose=True)


