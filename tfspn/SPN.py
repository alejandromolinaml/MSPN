'''
Created on Jan 5, 2017

@author: alejandro molina
@author: antonio vergari
'''
from collections import deque, OrderedDict
import gzip
import logging
import pickle
from platform import node

from numpy import float64
import numpy
from sklearn.cluster.dbscan_ import DBSCAN
from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.text import TfidfTransformer

from mlutils.LSH import above, make_planes
from mlutils.benchmarks import Chrono
from mlutils.stabilityTest import getIndependentGroupsStabilityTest
from mlutils.statistics import get_local_minimum
from pdn.ABPDN import ABPDN
from pdn.independenceptest import getIndependentGroups, getIndependentRDCGroups
from pdn.mixedClustering import getMixedGowerClustering
from tfspn.histogram import getHistogramVals
from tfspn.piecewise import piecewise_linear_approximation, estimate_bins, \
    estimate_domains, estimate_domains_range, compute_histogram
from tfspn.rdc import getIndependentGDTGroups_py
from tfspn.rdc import getIndependentRDCGroups_py, rdc_transformer
from tfspn.tfspn import SumNode, ProductNode, PoissonNode, GaussianNode, BernoulliNode
from tfspn.tfspn import CategoricalNode
from tfspn.tfspn import PiecewiseLinearPDFNodeOld, PiecewiseLinearPDFNode, IsotonicUnimodalPDFNode, HistNode, KernelDensityEstimatorNode
from tfspn.piecewise import compute_histogram_type_wise
from tfspn.piecewise import piecewise_linear_to_unimodal_isotonic
from tfspn.piecewise import histogram_to_piecewise_linear_type_wise


numpy.set_printoptions(threshold=numpy.inf)


class DataSlice:
    id = 0

    def __init__(self, data, families, domains, featureNames, featureTypes, rows, cols, noClusters=False, noIndependencies=False):
        self.id = DataSlice.id
        DataSlice.id += 1
        self._data = data
        self._rows = rows
        self._numInstances = len(rows)

        assert self._numInstances > 0, "no instances"

        self._cols = cols
        self._numFeatures = len(cols)

        assert self._numFeatures > 0, "no features"

        assert len(families) == data.shape[1], "invalid length of families for features"

        assert len(domains) == data.shape[1], "invalid length of families for features"

        assert len(featureTypes) == data.shape[1], "invalid length of families for features"

        assert len(featureNames) == data.shape[1], "invalid length of families for features"

        self._families = families
        self._domains = domains
        self._featureNames = featureNames
        self._featureTypes = numpy.array(featureTypes)

        self.noClusters = noClusters
        self.noIndependencies = noIndependencies

        self.nodeType = None
        self._parentSlice = None
        self._childrenSlices = []

    @property
    def numInstances(self):
        return self._numInstances

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def numFeatures(self):
        return self._numFeatures

    @property
    def families(self):
        return self._families[self._cols]

    @property
    def family(self):
        assert len(self._cols) == 1, "invalid number of columns != 1"
        return self._families[self._cols[0]]

    @property
    def domains(self):
        return self._domains

    @property
    def domain(self):
        assert len(self._cols) == 1, "invalid number of columns != 1"
        return self._domains[self._cols[0]]

    @property
    def featureTypes(self):
        return self._featureTypes

    @property
    def featureNames(self):
        return self._featureNames

    @property
    def featureLocalTypes(self):
        return self._featureTypes[self._cols]

    @property
    def featureName(self):
        assert len(self._cols) == 1, "invalid number of columns != 1"
        return self._featureNames[self._cols[0]]

    @property
    def featureType(self):
        assert len(self._cols) == 1, "invalid number of columns != 1"
        return self._featureTypes[self._cols[0]]

    @property
    def featureIdx(self):
        assert len(self._cols) == 1, "invalid number of columns != 1"
        return self._cols[0]

    @property
    def parentSlice(self):
        return self._parentSlice

    @parentSlice.setter
    def parentSlice(self, parent):
        self._parentSlice = parent

    def getData(self):
        return self._data[self._rows, :][:, self._cols]

    def getOHEData(self):
        cols = []
        for f in self._cols:
            cols.append(self.getOHEFeatureData(f))

        data = numpy.column_stack(cols)

        return data

    def getFeatureData(self, f):
        if f not in self._cols:
            raise Exception('invalid feature position %s' % (f))

        data = self._data[self._rows, :][:, f]

        return data

    def getOHEFeatureData(self, f):
        if f not in self._cols:
            raise Exception('invalid feature position %s' % (f))

        data = self._data[self._rows, :][:, f]

        if self._featureTypes[f] == 'categorical':
            # do one hot encoding of the feature
            domain = self._domains[f]

            dataenc = numpy.zeros((data.shape[0], len(domain)))

            dataenc[data[:, None] == domain[None, :]] = 1

            assert numpy.all((numpy.sum(dataenc, axis=1) == 1)), "one hot encoding bug"

            data = dataenc

        return data

    def getSliceIndexes(self, rowIndex=None, colIndex=None, noClusters=False, noIndependencies=False):

        rows = self._rows
        cols = self._cols

        if rowIndex is not None:
            rows = rows[rowIndex]

        if colIndex is not None:
            cols = cols[colIndex]

        return DataSlice(self._data, self._families, self._domains, self._featureNames, self._featureTypes, rows, cols, noClusters=noClusters, noIndependencies=noIndependencies)

    def getSlice(self, rows=None, cols=None, noClusters=False, noIndependencies=False):

        if rows is None:
            rows = self._rows

        if cols is None:
            cols = self._cols

        return DataSlice(self._data, self._families, self._domains, self._featureNames, self._featureTypes, rows, cols, noClusters=noClusters, noIndependencies=noIndependencies)

    def getBins(self):
        return self.data[self.rows, :][:, self.cols]

    def getMean(self):
        return numpy.mean(self.getData())

    def getSuccessRatio(self):

        data = self.getData()

        assert numpy.sum(numpy.unique(data)) <= 1, "data not binary"

        return numpy.sum(data) / data.shape[0]

    def getStdev(self):
        return numpy.std(self.getData())

    def addChildrenSlices(self, slices):
        self._childrenSlices.extend(slices)

    def getChildrenSlices(self):
        return self._childrenSlices

    def __repr__(self):
        return "DataSlice(%d, %s, %s, %s)\n" % (self.id, self.families, self.rows, self.cols)


class Splitting():

    @staticmethod
    def __GetClusteredDataSlices(data_slice, clusters, rows=True):
        unique_clusters = numpy.unique(clusters)

        num_clusters = len(unique_clusters)
        print('\t\t# clusters found', num_clusters)

        result = []

        if num_clusters == 1:
            return result, 1

        for uc in unique_clusters:
            if rows:
                result.append(data_slice.getSliceIndexes(rowIndex=clusters == uc))
            else:
                result.append(data_slice.getSliceIndexes(colIndex=clusters == uc))

            # print('\t# clusters found', num_clusters)
        return result, num_clusters

    @staticmethod
    def __preptfidf(data):
        tfidf_transformer = TfidfTransformer()
        return tfidf_transformer.fit_transform(data)

    @staticmethod
    def __preplog(data):
        return numpy.log(data + 1)

    @staticmethod
    def __prepsqrt(data):
        return numpy.sqrt(data)

    @staticmethod
    def GetFunction(name, config):
        if name == "kmeans":
            poissonPrep = config["poissonPrep"]
            n_clusters = config["n_clusters"]
            seed = config["seed"]
            ohe = config["OHE"]

            def clusteringFunctionKmeans(data_slice):

                if ohe:
                    data = data_slice.getOHEData()
                else:
                    data = data_slice.getData()

                if poissonPrep and "poisson" in data_slice.families:
                    if poissonPrep == "tf-idf":
                        f = Splitting._Splitting__preptfidf
                    elif poissonPrep == "log+1":
                        f = Splitting._Splitting__preplog
                    elif poissonPrep == "sqrt":
                        f = Splitting._Splitting__prepsqrt

                    data[:, data_slice.families == "poisson"] = f(
                        data[:, data_slice.families == "poisson"])

                clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionKmeans

        if name == "kmeansRDC":
            k = config["k"]
            s = config["s"]
            n_clusters = config["n_clusters"]
            ohe = config["OHE"]
            rand_gen = numpy.random.RandomState(config["seed"])

            def clusteringFunctionKmeansOHERDC(data_slice):

                logging.debug('Clustering by RDC K-means on slice {} ({}x{})'.format(data_slice.id,
                                                                                     len(data_slice.rows),
                                                                                     len(data_slice.cols)))

                data = rdc_transformer(data_slice,
                                       k=k,
                                       s=s,
                                       non_linearity=numpy.sin,
                                       return_matrix=True,
                                       ohe=ohe,
                                       rand_gen=rand_gen)

                clusters = KMeans(
                    n_clusters=n_clusters, random_state=rand_gen, n_jobs=1).fit_predict(data)

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionKmeansOHERDC

        if name == "DBScan":
            eps = config["eps"]
            min_samples = config["min_samples"]
            ohe = config["OHE"]

            def clusteringFunctionDBScanOHE(data_slice):

                if ohe:
                    data = data_slice.getOHEData()
                else:
                    data = data_slice.getData()

                clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionDBScanOHE

        if name == "Gower":
            n_clusters = config["n_clusters"]
            seed = config["seed"]

            def clusteringFunctionGower(data_slice):

                data = data_slice.getData()

                clusters = getMixedGowerClustering(
                    data, data_slice.featureLocalTypes, n_clusters, seed)

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionGower

        if name == "Random Partition":
            ohe = config["OHE"]

            def clusteringFunctionRandomPartition(data_slice):
                if ohe:
                    data = data_slice.getOHEData()
                else:
                    data = data_slice.getData()

                clusters = above(make_planes(1, data_slice.numFeatures), data)[:, 0]

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionRandomPartition

        if name == "Random Balanced Binary Split":

            rand_gen = numpy.random.RandomState(config["seed"])

            def clusteringFunctionRandomBinarySplit(data_slice):

                logging.debug('Clustering by a balanced random split')
                n_instances = data_slice.numInstances
                clusters = numpy.zeros(n_instances, dtype=int)
                clusters[:-(n_instances // 2)] = 1
                rand_gen.shuffle(clusters)
                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionRandomBinarySplit

        if name == "Random Binary Conditioning Split":

            rand_gen = numpy.random.RandomState(config["seed"])

            def clusteringFunctionRandomBinaryConditioningSplit(data_slice):

                logging.debug('Clustering by binary random conditioning')

                feature_types = numpy.array(data_slice.featureTypes)[data_slice.cols]
                #
                # domains shall be REestimated according to the current data_slice
                domains = estimate_domains(data_slice.getData(), feature_types)
                # logging.debug('estimated domains {}'.format(domains))

                #
                # select a random feature (relative indexing in slice columns)
                # rand_feature = rand_gen.choice(data_slice.cols)
                rand_feature = rand_gen.choice(len(data_slice.cols))
                # logging.debug('random feature: {} {} {}'.format(rand_feature, feature_types[rand_feature], domains[rand_feature]))

                #
                # randomly select a split point in its domain
                rand_split = rand_gen.choice(domains[rand_feature])
                # logging.debug('random split: {}'.format(rand_split))

                f_type = feature_types[rand_feature]
                data = data_slice.getOHEFeatureData(data_slice.cols[rand_feature])
                clusters = None
                if f_type == 'categorical':
                    clusters = data[:, rand_split]
                elif f_type == 'discrete':
                    clusters = (data == rand_split)
                elif f_type == 'continuous':
                    clusters = (data <= rand_split)

                # logging.debug('clusters {}'.format(clusters))

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionRandomBinaryConditioningSplit

        if name == "Random Binary Modal Splitting":

            rand_gen = numpy.random.RandomState(config["seed"])

            def clusteringFunctionRandomBinaryModalSplit(data_slice):

                logging.debug('Clustering by binary modal conditioning')

                feature_types = numpy.array(data_slice.featureTypes)[data_slice.cols]
                #
                # domains shall be REestimated according to the current data_slice
                domains = estimate_domains(data_slice.getData(), feature_types)
                # logging.debug('estimated domains {}'.format(domains))

                #
                # select a random feature (relative indexing in slice columns)
                # rand_feature = rand_gen.choice(data_slice.cols)
                rand_feature = rand_gen.choice(len(data_slice.cols))
                # logging.debug('random feature: {} {} {}'.format(rand_feature, feature_types[rand_feature], domains[rand_feature]))

                #
                # estimate the density
                data = data_slice.getFeatureData(data_slice.cols[rand_feature])
                density, bins = compute_histogram(data, bins=domains[rand_feature])

                #
                # get a (random if more than one) local minima
                lm = get_local_minimum(density, rand_gen=rand_gen)

                if lm is not None:
                    rand_split = domains[rand_feature][lm]
                    # print('\n\n\n\local minima', rand_split, '\n\n\n\n\n')
                else:
                    #
                    # going with random split
                    # print('\n\n\n>>>>>>>>>>> no local minima random split\n\n\n\n')
                    rand_split = rand_gen.choice(domains[rand_feature])
                    # logging.debug('random split: {}'.format(rand_split))

                # f_type = feature_types[rand_feature]
                # data = data_slice.getOHEFeatureData(data_slice.cols[rand_feature])
                # clusters = None
                # if f_type == 'categorical':
                #     clusters = data[:, rand_split]
                # elif f_type == 'discrete':
                #     clusters = (data == rand_split)
                # elif f_type == 'continuous':
                clusters = (data <= rand_split)

                # logging.debug('clusters {}'.format(clusters))

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionRandomBinaryModalSplit

        if name == "PDN":
            maxIters = config["maxIters"]
            max_depth = config["max_depth"]

            def clusteringFunctionPDN(data_slice):
                clusters = ABPDN.pdnClustering(
                    data_slice.getData(), nM=n_clusters, maxIters=maxIters, max_depth=max_depth)

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=True)
            return clusteringFunctionPDN

        if name == "IndependenceTest":
            alpha = config["alpha"]

            def clusteringFunctionIndependenceTest(data_slice):
                allowedFamilies = ["poisson", "gaussian", "bernoulli", "categorical", "baseline"]
                for fam in data_slice.families:
                    assert fam in allowedFamilies, "invalid family " + fam

                if numpy.all(data_slice.families == "poisson1"):
                    clusters = getIndependentGroupsStabilityTest(data_slice.getData(), alpha=alpha)
                else:
                    families = list(
                        map(lambda f: "binomial" if f == "bernoulli" else f, data_slice.families))
                    clusters = getIndependentGroups(
                        data_slice.getData(), alpha=alpha, families=families)
                    clusters = clusters.astype("int")

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=False)

            return clusteringFunctionIndependenceTest

        if name == "RDCTest":
            threshold = config["threshold"]
            ohe = config["OHE"]
            linear = config["linear"]

            def clusteringFunctionRDC(data_slice):

                logging.debug('Splitting columns by RDC on slice {} ({}x{})'.format(data_slice.id,
                                                                                    len(data_slice.rows),
                                                                                    len(data_slice.cols)))
                clusters = getIndependentRDCGroups(
                    data_slice.getData(), threshold, ohe, data_slice.featureLocalTypes, linear)

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=False)

            return clusteringFunctionRDC

        if name == "RDCTestpy":
            threshold = config["threshold"]

            def clusteringFunctionRDCpy(data_slice):
                clusters = getIndependentRDCGroups_py(data_slice, threshold)

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=False)

            return clusteringFunctionRDCpy

        if name == "GDTTest":
            threshold = config["threshold"]

            def clusteringFunctionGDT(data_slice):
                clusters = getIndependentGDTGroups_py(data_slice, threshold)

                return Splitting._Splitting__GetClusteredDataSlices(data_slice, clusters, rows=False)

            return clusteringFunctionGDT

        assert False, "Invalid clustering name: " + name

    @staticmethod
    def KmeansRows(n_clusters=2, OHE=True, seed=17):
        config = {'poissonPrep': "log+1", 'n_clusters': n_clusters, 'seed': seed, "OHE": OHE}
        name = "kmeans"
        return (name, config)

    @staticmethod
    def KmeansRDCRows(n_clusters=2, k=10, s=1 / 6, OHE=True, seed=17):
        config = {'n_clusters': n_clusters, 'seed': seed, "OHE": OHE, 'k': k, 's': s}
        name = "kmeansRDC"
        return (name, config)

    @staticmethod
    def Gower(n_clusters=2, seed=17):
        config = {'n_clusters': n_clusters, 'seed': seed}
        name = "Gower"
        return (name, config)

    @staticmethod
    def DBScan(eps=0.3, min_samples=10, OHE=True):
        config = {'eps': eps, 'min_samples': min_samples, "OHE": OHE}
        name = "DBScanOHE"
        return (name, config)

    @staticmethod
    def RandomPartitionRows(OHE=True):
        config = {"OHE": OHE}
        name = "Random Partition"
        return (name, config)

    @staticmethod
    def RandomBalancedBinarySplit(seed=17):
        config = {'seed': seed}
        name = "Random Balanced Binary Split"
        return (name, config)

    @staticmethod
    def RandomPartitionConditioningRows(seed=17):
        config = {'seed': seed}
        name = "Random Binary Conditioning Split"
        return (name, config)

    @staticmethod
    def RandomBinaryModalSplitting(seed=17):
        config = {'seed': seed}
        name = "Random Binary Modal Splitting"
        return (name, config)

    @staticmethod
    def ClusteringPDNRows(n_clusters=2, maxIters=5, max_depth=5):
        config = {'maxIters': maxIters, 'n_clusters': n_clusters, 'max_depth': max_depth}
        name = "PDN"
        return (name, config)

    @staticmethod
    def IndependenceTest(alpha=0.001):
        config = {'alpha': alpha}
        name = "IndependenceTest"
        return (name, config)

    @staticmethod
    def RDCTest(threshold=0.3, OHE=True, linear=False):
        config = {'threshold': threshold, 'linear': linear, 'OHE': OHE}
        name = "RDCTest"
        return (name, config)

    @staticmethod
    def RDCTestOHEpy(threshold=0.3):
        config = {'threshold': threshold}
        name = "RDCTestpy"
        return (name, config)

    @staticmethod
    def GDTTest(threshold=0.3):
        config = {'threshold': threshold}
        name = "GDTTest"
        return (name, config)


class SPN(object):

    '''
    classdocs
    '''

    def __init__(self, **kwargs):
        self.config = dict(kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.current_id = 0
        self.root = None

    def getNextId(self):
        res = self.current_id
        self.current_id += 1
        return res

    def size(self):
        return self.root.size()

    def findOperation(self, data_slice, cluster_first):
        minimalFeatures = data_slice.numFeatures == 1
        minimalInstances = data_slice.numInstances <= self.min_instances_slice

        if minimalFeatures:
            return "CreateLeaf"

        if minimalInstances or (data_slice.noClusters and data_slice.noIndependencies):
            return "NaiveFactorization"

        if data_slice.parentSlice is None and not data_slice.noClusters:
            print("ROOT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

            return "Cluster" if cluster_first else "SplitFeatures"

        if data_slice.noIndependencies:
            print(
                "NO INDEPENDENCIES FOUND #####################################################################################")
            return "Cluster"

        print(
            "SEARCHING INDEPENDENCIES &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        return "SplitFeatures"

    def findOperation2(self, data_slice, cluster_first, cluster_univariate=False):
        #         Features=1    Instances=1    noIndependencies    noClusters        OP
        #         0    0    0    0        FlipOp
        #         0    0    0    1        split features
        #         0    0    1    0        split clusters
        #
        #
        #
        #         0    0    1    1        naive factorization
        #
        #         0    1    0    0        naive factorization
        #         0    1    0    1        naive factorization
        #         0    1    1    0        naive factorization
        #         0    1    1    1        naive factorization
        #
        #
        #         1    0    0    0        split clusters
        #         1    0    0    1        leaf
        #         1    0    1    0        split clusters
        #         1    0    1    1        leaf
        #         1    1    0    0        leaf
        #         1    1    0    1        leaf
        #         1    1    1    0        leaf
        #         1    1    1    1        leaf

        minimalFeatures = data_slice.numFeatures == 1
        minimalInstances = data_slice.numInstances <= self.min_instances_slice

        if minimalFeatures:
            if minimalInstances or data_slice.noClusters:
                return "CreateLeaf"
            else:
                if cluster_univariate:
                    return "Cluster"
                else:
                    return "CreateLeaf"

        zeroVarianceVarsCount = numpy.sum(numpy.var(data_slice.getData(), 0) == 0)
        if zeroVarianceVarsCount > 1:
            if zeroVarianceVarsCount == data_slice.numFeatures:
                return "NaiveFactorization"
            else:
                return "removeUninformativeFeatures"

        if minimalInstances or (data_slice.noClusters and data_slice.noIndependencies):
            return "NaiveFactorization"

        if data_slice.noIndependencies:
            return "Cluster"

        if data_slice.noClusters:
            return "SplitFeatures"

        if data_slice.parentSlice is None:
            return "Cluster" if cluster_first else "SplitFeatures"

        # this can be done to train a deeper model
        # if data_slice.numFeatures > data_slice.numInstances:
        #    return "SplitFeatures"
        # else:
        #    return "Cluster"
        return "SplitFeatures"

        # this can be done to alternate clustering and splitting
        # return "Cluster" if data_slice.parentSlice.nodeType == ProductNode else "SplitFeatures"

    @staticmethod
    def LearnStructure(X,
                       featureTypes,
                       domains=None,
                       families=None,
                       featureNames=None,
                       min_instances_slice=200,
                       bin_width=1,
                       alpha=1,
                       isotonic=False,
                       pw_bootstrap=None,
                       avg_pw_boostrap=False,
                       row_split_method=Splitting.KmeansRows(),
                       col_split_method=Splitting.IndependenceTest(),
                       cluster_first=True,
                       prior_weight=0.01,
                       kernel_family='gaussian',
                       kernel_bandwidth=0.2,
                       kernel_metric='euclidean',
                       rand_seed=17):

        numpy.random.seed(rand_seed)

        assert row_split_method is not None, "row_split_method not set"
        assert col_split_method is not None, "col_split_method not set"
        assert featureNames is None or len(
            featureNames) == X.shape[1], "wrong number of featureNames"

        if domains is None:
            domains = estimate_domains_range(X, featureTypes)

        data = X.astype(float64)

        print(data.shape)

        if featureNames is None:
            featureNames = ["X_%s_" % (i) for i in range(data.shape[1])]

        if isinstance(families, str):
            families = [families] * data.shape[1]

        if families is None:
            families = ['piecewise'] * data.shape[1]

        spn = SPN(min_instances_slice=min_instances_slice, row_split_method=row_split_method,
                  col_split_method=col_split_method, families=families, domains=domains, featureTypes=featureTypes, featureNames=featureNames, numFeatures=data.shape[1])

        rootSlice = DataSlice(data, numpy.asarray(families), domains, featureNames, featureTypes, numpy.arange(
            data.shape[0]), numpy.arange(data.shape[1]))

        data_slices = deque()
        data_slices.append(rootSlice)

        cluster_univariate = True if 'isotonic' in families else False
        print('CLUSTER UNIVARIATE:{}\n'.format(cluster_univariate))

        while data_slices:
            data_slice = data_slices.popleft()

            operation = spn.findOperation2(data_slice, cluster_first, cluster_univariate)

            print("slices to go: ", len(data_slices), "operation", operation)

            #print(operation, data_slice.id,                  "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            # print(operation, data_slice, "no clusters", data_slice.noClusters, "no independencies", data_slice.noIndependencies)

            if operation == "removeUninformativeFeatures":
                variances = numpy.var(data_slice.getData(), axis=0)

                cols = []
                for c, col in enumerate(data_slice.cols):
                    if variances[c] == 0:
                        # IF no variance, then split it appart and set no clustering to true
                        new_data_slice = data_slice.getSlice(
                            cols=numpy.asarray([col]), noClusters=True)
                        new_data_slice.parentSlice = data_slice
                        data_slice.addChildrenSlices([new_data_slice])
                        data_slices.append(new_data_slice)

                        logging.debug('\tfeature {} ({}) with {} var'.format(col,
                                                                             data_slice.featureNames[
                                                                                 col],
                                                                             variances[c]))
                    else:
                        cols.append(col)

                # all the other features with data, should be joined
                new_data_slice = data_slice.getSlice(cols=numpy.asarray(cols))
                new_data_slice.parentSlice = data_slice
                data_slice.addChildrenSlices([new_data_slice])
                data_slices.append(new_data_slice)

                data_slice.nodeType = ProductNode

                continue

            elif operation == "Cluster":

                c = Chrono().start()
                new_data_slices, nr_splits = Splitting.GetFunction(*row_split_method)(data_slice)
                # print("Running clustering in", c.end().elapsed(), "seconds")

                if nr_splits == 1:
                    data_slice.noClusters = True
                    data_slices.append(data_slice)
                    continue

                for nds in new_data_slices:
                    nds.parentSlice = data_slice

                data_slice.addChildrenSlices(new_data_slices)

                data_slice.nodeType = SumNode
                data_slices.extend(new_data_slices)

                continue

            elif operation == "SplitFeatures":
                c = Chrono().start()
                new_data_slices, nr_splits = Splitting.GetFunction(*col_split_method)(data_slice)
                # print("Running independency test in", c.end().elapsed(), "seconds")

                if nr_splits == 1:
                    data_slice.noIndependencies = True
                    data_slices.append(data_slice)
                    continue

                for nds in new_data_slices:
                    nds.parentSlice = data_slice

                data_slice.addChildrenSlices(new_data_slices)

                data_slice.nodeType = ProductNode
                data_slices.extend(new_data_slices)

                continue

            elif operation == "NaiveFactorization":

                for c, col in enumerate(data_slice.cols):
                    new_data_slice = data_slice.getSlice(cols=numpy.asarray([col]))
                    new_data_slice.parentSlice = data_slice
                    data_slice.addChildrenSlices([new_data_slice])
                    data_slices.append(new_data_slice)
                data_slice.nodeType = ProductNode

                continue

            elif operation == "CreateLeaf":

                if data_slice.family == 'baseline':
                    ftype = data_slice.featureType
                    if ftype == 'continuous':
                        data_slice.nodeType = GaussianNode
                    elif ftype in {'categorical', 'discrete'}:
                        data_slice.nodeType = CategoricalNode
                    else:
                        raise ValueError('Unrecognized feature type, {}'.format(ftype))

                elif data_slice.family in PoissonNode.families:
                    data_slice.nodeType = PoissonNode

                elif data_slice.family in GaussianNode.families:
                    data_slice.nodeType = GaussianNode

                elif data_slice.family in BernoulliNode.families:
                    data_slice.nodeType = BernoulliNode

                elif data_slice.family in PiecewiseLinearPDFNodeOld.families:
                    data_slice.nodeType = PiecewiseLinearPDFNodeOld

                elif data_slice.family in PiecewiseLinearPDFNode.families:
                    data_slice.nodeType = PiecewiseLinearPDFNode

                elif data_slice.family in IsotonicUnimodalPDFNode.families:
                    data_slice.nodeType = IsotonicUnimodalPDFNode

                elif data_slice.family in HistNode.families:
                    data_slice.nodeType = HistNode

                elif data_slice.family in KernelDensityEstimatorNode.families:
                    data_slice.nodeType = KernelDensityEstimatorNode

                else:
                    raise Exception('Invalid family: ' + data_slice.family)

            else:
                raise Exception('Invalid operation: ' + operation)

        # take dataslices and build spn
        spn.root = spn.BuildSpn(rootSlice,
                                bin_width=bin_width,
                                alpha=alpha,
                                isotonic=isotonic,
                                pw_bootstrap=pw_bootstrap,
                                avg_pw_boostrap=avg_pw_boostrap,
                                prior_weight=prior_weight,
                                kernel_family=kernel_family,
                                kernel_bandwidth=kernel_bandwidth,
                                kernel_metric=kernel_metric)

        # prune spn
        spn.root.Prune()

        return spn

    @staticmethod
    def FromFile(fname):
        import jsonpickle
        import gzip
        import jsonpickle.ext.numpy as jsonpickle_numpy
        jsonpickle_numpy.register_handlers()

        with gzip.GzipFile(filename=fname, mode="rb") as f:
            return jsonpickle.decode(f.read().decode())

    def ToFile(self, fname):
        import jsonpickle
        import gzip
        import jsonpickle.ext.numpy as jsonpickle_numpy
        jsonpickle_numpy.register_handlers()

        with gzip.GzipFile(filename=fname, mode="w") as f:
            return f.write(jsonpickle.encode(self).encode())

    @staticmethod
    def to_pickle(spn, fname):

        with gzip.open(fname, mode="wb") as f:
            pickle.dump(spn, f, protocol=-1)

    @staticmethod
    def from_pickle(fname):

        with gzip.open(fname, mode='rb') as f:
            return pickle.load(f)

    def __getstate__(self):
        return {"config": self.config, "current_id": self.current_id, "root": self.root}

    def BuildSpn(self,
                 data_slice,
                 bin_width=1,
                 alpha=1,
                 isotonic=True,
                 pw_bootstrap=None,
                 avg_pw_boostrap=False,
                 prior_weight=0.01,
                 kernel_family='gaussian',
                 kernel_bandwidth=0.2,
                 kernel_metric='euclidean'):

        if data_slice.nodeType == KernelDensityEstimatorNode:

            prior_density = None
            if data_slice.featureType == 'continuous':
                prior_density = 1 / (data_slice.domain.max() - data_slice.domain.min())
            elif data_slice.featureType in {'discrete', 'categorical'}:
                prior_density = 1 / len(data_slice.domain)

            node = KernelDensityEstimatorNode(name='KDENode_{}'.format(self.getNextId()),
                                              featureIdx=data_slice.featureIdx,
                                              featureName=data_slice.featureName,
                                              data=data_slice.getData(),
                                              domain=data_slice.domain,
                                              prior_weight=prior_weight,
                                              prior_density=prior_density,
                                              kernel=kernel_family,
                                              bandwidth=kernel_bandwidth,
                                              metric=kernel_metric)

            KernelDensityEstimatorNode.check_eval(node, n_features=data_slice._data.shape[1])

        elif data_slice.nodeType == PiecewiseLinearPDFNodeOld:

            prior_density = None
            if data_slice.featureType == 'continuous':
                prior_density = 1 / (data_slice.domain.max() - data_slice.domain.min())
            elif data_slice.featureType in {'discrete', 'categorical'}:
                prior_density = 1 / len(data_slice.domain)

            # print("estimate_bins")

            # print(data_slice.domain)
            nb = 10
            bins = estimate_bins(data_slice.getData(), data_slice.featureType, [data_slice.domain])
            # bins = ['auto']
            # print(bins)
            if pw_bootstrap is not None:
                bins = bins * pw_bootstrap
            # print(bins)
            # print('bins', data_slice.featureType, data_slice.featureIdx, bins)
            # print('domains', data_slice.domain)
            # print('bins', bins)
            # bins = estimate_bins(data_slice.getData(), data_slice.featureType, data_slice.domain)
            # print("piecewise_linear_approximation", data_slice.getData().shape, data_slice.featureType)
            # x_range, y_range = piecewise_linear_approximation(
            #     data_slice.getData(),
            #     bins,
            #     # bins * nb,
            #     data_slice.featureType, alpha=1,
            #     # isotonic=True,
            #     # n_bootstraps=nb, average_bootstraps=False
            # )
            # print("PiecewiseLinearPDFNode")
            # node = PiecewiseLinearPDFNode("PiecewiseLinearNode_" + str(
            # self.getNextId()), data_slice.featureIdx, data_slice.featureName,
            # data_slice.domain, x_range, y_range)

            x_range, y_range = piecewise_linear_approximation(data_slice.getData(),
                                                              bins=bins,
                                                              family=data_slice.featureType,
                                                              bin_width=bin_width,
                                                              alpha=alpha,
                                                              isotonic=isotonic,
                                                              n_bootstraps=pw_bootstrap,
                                                              average_bootstraps=avg_pw_boostrap,)

            node = PiecewiseLinearPDFNode("PiecewiseLinearNodeOLD_" + str(self.getNextId()),
                                          data_slice.featureIdx,
                                          data_slice.featureName,
                                          data_slice.domain,
                                          x_range, y_range,
                                          prior_weight=prior_weight,
                                          prior_density=prior_density,)
            # print(node)
            # print("node done")

        elif data_slice.nodeType == PiecewiseLinearPDFNode:

            #
            # first the the histogram
            densities, breaks, prior_density, bin_repr_points = compute_histogram_type_wise(data=data_slice.getData(),
                                                                                            feature_type=data_slice.featureType,
                                                                                            domain=data_slice.domain,
                                                                                            alpha=alpha)

            # print('DENSITIES', densities)
            # print('BINS', breaks)
            # print('PRIOR', prior_density)

            #
            # then a piecewise linear
            x, y = histogram_to_piecewise_linear_type_wise(densities,
                                                           bins=breaks,
                                                           feature_type=data_slice.featureType)

            node = PiecewiseLinearPDFNode("PiecewiseLinearNode_" + str(self.getNextId()),
                                          featureIdx=data_slice.featureIdx,
                                          featureName=data_slice.featureName,
                                          domain=data_slice.domain,
                                          x_range=x, y_range=y,
                                          prior_weight=prior_weight,
                                          prior_density=prior_density,
                                          bin_repr_points=bin_repr_points)
            #node.rows = data_slice.rows()

        elif data_slice.nodeType == IsotonicUnimodalPDFNode:

            #
            # first the the histogram
            densities, breaks, prior_density, bin_repr_points = compute_histogram_type_wise(data=data_slice.getData(),
                                                                                            feature_type=data_slice.featureType,
                                                                                            domain=data_slice.domain,
                                                                                            alpha=alpha)
            #
            # then a piecewise linear
            x, y = histogram_to_piecewise_linear_type_wise(densities,
                                                           bins=breaks,
                                                           feature_type=data_slice.featureType)

            #
            # isotonic
            x, y = piecewise_linear_to_unimodal_isotonic(x, y)

            node = IsotonicUnimodalPDFNode("IsotonicUnimodalNode_" + str(self.getNextId()),
                                           featureIdx=data_slice.featureIdx,
                                           featureName=data_slice.featureName,
                                           domain=data_slice.domain,
                                           x_range=x, y_range=y,
                                           prior_weight=prior_weight,
                                           prior_density=prior_density,
                                           bin_repr_points=bin_repr_points)

        elif data_slice.nodeType == HistNode:

            densities, breaks, prior_density, bin_repr_points = compute_histogram_type_wise(data=data_slice.getData(),
                                                                                            feature_type=data_slice.featureType,
                                                                                            domain=data_slice.domain,
                                                                                            alpha=alpha)

            node = HistNode("HistNode_" + str(self.getNextId()), data_slice.featureIdx,
                            data_slice.featureName, breaks,
                            densities, prior_density, prior_weight=prior_weight,
                            bin_repr_points=bin_repr_points)

        elif data_slice.nodeType == CategoricalNode:
            #
            # estimate data by counting
            densities, breaks, _prior_density, _bin_repr_points = compute_histogram_type_wise(data=data_slice.getData(),
                                                                                              feature_type=data_slice.featureType,
                                                                                              domain=data_slice.domain,
                                                                                              alpha=alpha)

            node = CategoricalNode("CategoricalNode_" + str(self.getNextId()),
                                   data_slice.featureIdx,
                                   data_slice.featureName,
                                   densities)

        elif data_slice.nodeType == PoissonNode:
            node = PoissonNode("PoissonNode_" + str(self.getNextId()),
                               data_slice.featureIdx, data_slice.featureName, data_slice.getMean())

        elif data_slice.nodeType == GaussianNode:
            node = GaussianNode("GaussianNode_" + str(self.getNextId()), data_slice.featureIdx,
                                data_slice.featureName, data_slice.getMean(), data_slice.getStdev())

        elif data_slice.nodeType == BernoulliNode:
            node = BernoulliNode("BernoulliNode_" + str(self.getNextId()),
                                 data_slice.featureIdx, data_slice.featureName, data_slice.getSuccessRatio())

        elif data_slice.nodeType == ProductNode:
            node = ProductNode("ProductNode_" + str(self.getNextId()))
            for c in data_slice.getChildrenSlices():
                node.addChild(self.BuildSpn(c, bin_width=bin_width,
                                            alpha=alpha,
                                            isotonic=isotonic,
                                            pw_bootstrap=pw_bootstrap,
                                            avg_pw_boostrap=avg_pw_boostrap,
                                            prior_weight=prior_weight,
                                            kernel_family=kernel_family,
                                            kernel_bandwidth=kernel_bandwidth,
                                            kernel_metric=kernel_metric))
            node.rows = data_slice.rows

        elif data_slice.nodeType == SumNode:
            node = SumNode("SumNode_" + str(self.getNextId()))
            for c in data_slice.getChildrenSlices():
                node.addChild(float(c.numInstances) / data_slice.numInstances, self.BuildSpn(c,
                                                                                             bin_width=bin_width,
                                                                                             alpha=alpha,
                                                                                             isotonic=isotonic,
                                                                                             pw_bootstrap=pw_bootstrap,
                                                                                             avg_pw_boostrap=avg_pw_boostrap,
                                                                                             prior_weight=prior_weight,
                                                                                             kernel_family=kernel_family,
                                                                                             kernel_bandwidth=kernel_bandwidth,
                                                                                             kernel_metric=kernel_metric))
            node.rows = data_slice.rows

        else:
            print(data_slice)
            raise Exception('Invalid node type ' + str(data_slice.nodeType))

        return node

    def marginalize(self, featureIds):

        features = [i for i in range(len(self.config["families"])) if (i not in featureIds)]

        margSPN = self.root.marginalizeOut(features)

        margSPN.Prune()

        return margSPN

    def to_graph(self):
        import networkx as nx
        G = nx.DiGraph()

        rootNode = self.root

        G.add_node(rootNode.name, label=rootNode.label)

        nodes = [rootNode]

        while(len(nodes) > 0):

            node = nodes.pop(0)

            for i, c in enumerate(node.children):

                G.add_node(c.name, label=c.label)

                weight = ""
                if hasattr(node, "weights"):
                    weight = round(node.weights[i], 2)

                G.add_edge(node.name, c.name, weight=1.0, label=weight)

                if c.children and len(c.children) > 0:
                    nodes.append(c)

        return G

    def save_pdf_graph(self, outputfile=None):

        if outputfile is None:
            return

        import networkx.drawing.nx_pydot as nxpd
        import tempfile
        import os.path
        from shutil import copyfile

        G = self.to_graph()
        pdG = nxpd.to_pydot(G)

        tmpoutputfile = tempfile.NamedTemporaryFile().name

        from PyPDF2 import PdfFileWriter, PdfFileReader
        import io
        from reportlab.pdfgen import canvas

        tf = tempfile.NamedTemporaryFile()
        pdG.write_pdf(tf.name)

        packet = io.BytesIO()
        can = canvas.Canvas(packet)

        can.drawString(0, 20, "Config: " + self.configStr())

        can.save()
        packet.seek(0)
        new_pdf = PdfFileReader(packet)

        existing_pdf = PdfFileReader(open(tf.name, "rb"))
        output = PdfFileWriter()

        page = existing_pdf.getPage(0)
        page2 = new_pdf.getPage(0)
        page.mergePage(page2)
        output.addPage(page)

        outputStream = open(tmpoutputfile, "wb")
        output.write(outputStream)
        outputStream.close()

        if os.path.isfile(outputfile):
            from PyPDF2 import PdfFileMerger
            merger = PdfFileMerger()
            for filename in [outputfile, tmpoutputfile]:
                merger.append(PdfFileReader(open(filename, 'rb')))

            merger.write(outputfile)

        else:
            copyfile(tmpoutputfile, outputfile)

    def top_down_nodes(self):
        nodes_to_process = deque()
        nodes_to_process.append(self.root)
        visited_nodes = set()

        while nodes_to_process:
            n = nodes_to_process.popleft()
            if n not in visited_nodes:
                yield n
                nodes_to_process.extend(n.children)
                visited_nodes.add(n)

    def n_nodes(self):
        return len(list(self.top_down_nodes()))

    def n_edges(self):
        n_edges = 0
        for n in self.top_down_nodes():
            n_edges += len(n.children)
        return n_edges

    def get_nodes_by_type(self, nodeType):
        return [n for n in self.top_down_nodes() if isinstance(n, nodeType)]

    def n_sum_nodes(self):
        return len(self.get_nodes_by_type(SumNode))

    def n_prod_nodes(self):
        return len(self.get_nodes_by_type(ProductNode))

    def get_leaves(self):
        return [n for n in self.top_down_nodes() if len(n.children) == 0]

    def n_leaves(self):
        return len(self.get_leaves())

    def n_layers(self):
        n_layers = {self.root: 0}
        for n in self.top_down_nodes():
            depth = n_layers[n]
            for c in n.children:
                n_layers[c] = depth + 1

        return max(n_layers.values())

    def configStr(self):
        s = OrderedDict(self.config)
        #
        # quick hack to fix domain issues
        try:
            s['domains'] = list(d.tolist() for d in s['domains'])
            print('Domain information', s)
        except:
            print('No domain information', s)

        import json
        return json.dumps(s)

    def __repr__(self):
        return self.configStr() + "\n" + str(self.root)
