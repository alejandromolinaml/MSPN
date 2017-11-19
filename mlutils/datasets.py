import logging
import operator
import codecs
import datetime
import os
import re
from collections import Counter

from joblib import memory
from numpy import float64
import numpy

from tfspn.piecewise import compute_histogram
from tfspn.piecewise import BINNING_METHOD

path = os.path.dirname(__file__) + "/data/"


def getCIFAR10(grayscale=True):


    def tograyscale(imgs):
        result = numpy.zeros((imgs.shape[0], int(imgs.shape[1]/3)))

        for i in range(result.shape[0]):
            result[i, :] = imgs[i,0:1024] * 0.2989 + imgs[i,1024:2048] * 0.5870 + imgs[i,2048:] * 0.1140 #matlabs constants for RGB to Grayscale


        return result

    def getData(fnames):

        import pickle

        data = None
        labels = None
        for fname in fnames:
            with open(path+"cifar-10-batches-py/"+fname, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')

                if data is None:
                    data = dict[b'data']
                    labels = dict[b'labels']
                else:
                    data = numpy.vstack((data, dict[b'data']))
                    labels = labels + dict[b'labels']
        return (data, numpy.array(labels))

    train, labels_train = getData(["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"])
    print(train.shape, labels_train.shape)

    test, labels_test = getData(["test_batch"])
    print(test.shape, labels_test.shape)

    if grayscale:
        train = tograyscale(train)
        test = tograyscale(test)

    return ("Cifar10", train, test, labels_train.reshape(train.shape[0], 1), labels_test.reshape(test.shape[0], 1))




def getTraffic():
    fname = path + "/traffic.csv"
    words = str(open(fname, "rb").readline()).split(';')

    words = list(map(lambda w: w.replace('"', '').strip(), words))
    words = words[2:]

    D = numpy.loadtxt(fname, dtype="S20", delimiter=";", skiprows=1)
    times = D[:, 1]
    D = D[:, 2:]

    nas = numpy.zeros_like(D, dtype="S20")
    nas[:] = "NA"

    times = times[numpy.all(D != nas, axis=1)]
    D = D[numpy.all(D != nas, axis=1)]

    D = D.astype(float)

    hours = map(lambda t: float(datetime.datetime.fromtimestamp(
        int(t)
    ).strftime('%H')), times)

    hours = numpy.asmatrix(hours).T

    return (D, words, times, hours)


def getGrolier():

    words = list(map(lambda line: line.decode(encoding='UTF-8').strip(),
                     open(path + "grolier15276_words.txt", "rb").readlines()))

    documents = list(map(lambda line: line.decode(
        encoding='UTF-8').strip().replace(",,", ""), open(path + "grolier15276.csv", "rb").readlines()))

    D = numpy.zeros((len(documents), len(words)))

    for i, doc in enumerate(documents):
        doc = doc.split(",")[1:]
        for j in range(0, len(doc), 2):
            D[i, int(doc[j]) - 1] = int(doc[j + 1])

    return ("Grolier", D, words)


def getNips():
    fname = path + "nips100.csv"
    words = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    return ("Nips", D, words)

def getHydrochem():
    fname = path + "hydrochem.csv"
    words = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    return ("Hydrochem", D, words)

def getAirQualityUCITimeless():
    fname = path + "AirQualityUCITimeless.csv"
    features = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = numpy.loadtxt(fname, dtype=float, delimiter=";", skiprows=1)
    return ("AirQualityUCITimeless", D, features)


def getAdult():
    fname = path + "adult.data"
    D = numpy.loadtxt(fname, delimiter=",", skiprows=0, dtype="S")
    D = numpy.char.strip(D)

    D = D[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

    completeIdx = numpy.sum(D == b' ?', axis=1) == 0

    D = D[completeIdx, :]

    def isinteger(x):
        return numpy.all(numpy.equal(numpy.mod(x, 1), 0))

    cols = []
    types = []
    domains = []

    for col in range(D.shape[1]):
        b, c = numpy.unique(D[:, col], return_inverse=True)

        try:
            # could convert to float
            if isinteger(b.astype(float)):
                # was integer
                cols.append(D[:, col].astype(int))
                types.append("discrete")
                domains.append(b.astype(int))
                continue

            # was float
            cols.append(D[:, col].astype(float))
            types.append("continuous")
            domains.append(b.astype(float))

            continue
        except:
            # was discrete
            cols.append(c)
            types.append("categorical")
            domains.append(b.astype(str))

    data = numpy.column_stack(cols)

    types[-4] = "continuous"
    types[-5] = "continuous"

    features = ["Age", "Work Type", "Education Level", "Education Level #", "Marital Status", "Occupation",
                "Relationship", "Race", "Gender", "Capital Gain", "Capital Loss", "Hours Per Week", "Native Country", "Income"]

    return ("Adult", data, features, types, domains)


def getMSNBCclicks():
    fname = path + "MSNBC.pdn.csv"
    words = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    return ("MSNBC", D, words)


def getSynthetic():
    fname = path + "synthetic.csv"
    words = sum([["A" + str(i) for i in range(1, 51)], ["B" + str(i) for i in range(1, 51)]], [])

    D = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=0)
    return ("Synthetic", D, words)


def getCommunitiesAndCrimes(filtered=True):
    words = map(lambda x: x.decode(encoding='UTF-8').strip(),
                open(path + "communities_names.txt", "rb").readlines())
    words = [word.split() for word in words if word.startswith("@attribute")]

    D = numpy.loadtxt(path + "communities.txt", dtype='S8',
                      delimiter=",").view(numpy.chararray).decode('utf-8')

    if not filtered:
        return ("C&C", D, words)

    numidx = [i for i in range(len(words)) if words[i][2] == "numeric"]
    words = [words[i][1] for i in range(len(words)) if words[i][2] == "numeric"]

    D = D[:, numidx]

    words = [words[-18], words[-16], words[-14], words[-12],
             words[-10], words[-8], words[-6], words[-4]]
    D = D[:, (-18, -16, -14, -12, -10, -8, -6, -4)]

    denseidx = [r for r in range(D.shape[0]) if not any(D[r, :] == "?")]
    D = D[denseidx, :]

    D = D.astype(float)
    return ("C&C", D, words)


def getSpambase(instances=4601):
    words = map(lambda x: x.decode(encoding='UTF-8').strip(),
                open(path + "spambase_names.txt", "rb").readlines())
    words = [word.split() for word in words if word.startswith("@attribute")]

    D = numpy.loadtxt(
        path + "spambase.txt", dtype='S8', delimiter=",").view(numpy.chararray).decode('utf-8')

    numidx = [i for i in range(len(words)) if words[i][2] == "REAL"]
    words = [words[i][1] for i in range(len(words)) if words[i][2] == "REAL"]
    numidx.append(57)  # class column
    words.append('class')  # class column

    D = D[:instances, numidx]
    # D = D[1000:3000, numidx]
    D = D.astype(float)
    return ("Spambase", D, words)


def getIRIS(classes=2):
    assert classes == 2 or classes == 3
    from sklearn import datasets
    from sklearn.cross_validation import train_test_split

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    labels = iris.feature_names
    y = y.reshape(len(y), 1)
    data = numpy.hstack((y, X))
    data = data[:(classes * 50), :]  # 50 instances per class

    labels = numpy.append('class', labels)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label

    data = data.astype(float)
    numpy.random.seed(42)
    numpy.random.shuffle(data)
    return "IRIS" + str(classes), data, labels, classes, families


def getSyntheticClassification(classes, features, informative, samples):
    import sklearn.datasets
    X, y = sklearn.datasets.make_classification(
        n_samples=samples, n_features=features, n_informative=informative, n_classes=classes, random_state=42, shuffle=True)
    y = y.reshape(len(y), 1)
    data = numpy.hstack((y, X))
    labels = ['x' + str(i) for i in range(features)]
    labels.insert(0, 'class')

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label

    name = "SYN_" + str(classes) + '_' + str(features) + '_' + str(samples)

    return name, data, labels, classes, families


def getWisconsinBreastCancer():

    fname = path + "wisonsinbreastcancer.csv"

    labels = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    labels = list(map(lambda w: w.replace('"', '').strip(), labels))
    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    data = data.astype(float)

    # remove "id" column
    labels.remove('id')
    data = numpy.delete(data, 0, axis=1)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label
    return "BreastCancer", data, labels, 2, families


def getGlass():

    fname = path + "glass.csv"

    labels = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    labels = list(map(lambda w: w.replace('"', '').strip(), labels))

    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    data = data.astype(float)

    # remove "id" column
    labels.remove('ID')
    data = numpy.delete(data, 0, axis=1)

    # change class index to first column
    labels = numpy.roll(labels, 1)
    data = numpy.roll(data, 1, axis=1)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label

    # TODO refactor classes starting with 0
    return "Glass", data, labels, 6, families


def getDiabetes():

    fname = path + "pima-indians-diabetes.data.csv"

    labels = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    labels = list(map(lambda w: w.replace('"', '').strip(), labels))

    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    data = data.astype(float)

    # change class index to first column
    labels = numpy.roll(labels, 1)
    data = numpy.roll(data, 1, axis=1)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label
    return "Diabetes", data, labels, 2, families


def getIonosphere():
    fname = path + "ionosphere.data.csv"

    # bad = class 0
    # good = class 1
    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=0)
    data = data.astype(float)

    labels = ['x' + str(i) for i in range(data.shape[1])]
    labels[data.shape[1] - 1] = 'class'

    # remove x2 - always 0
    labels.remove('x1')
    data = numpy.delete(data, 1, axis=1)

    labels.remove('x0')
    data = numpy.delete(data, 0, axis=1)

    # change class index to first column
    labels = numpy.roll(labels, 1)
    data = numpy.roll(data, 1, axis=1)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label
    return "Ionosphere", data, labels, 2, families


def getWineQualityWhite():

    fname = path + "winequality-white.csv"

    labels = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(';')
    labels = list(map(lambda w: w.replace('"', '').strip(), labels))
    data = numpy.loadtxt(fname, dtype=float, delimiter=";", skiprows=1)
    data = data.astype(float)

    # change index to first column
    labels = numpy.roll(labels, 1)
    data = numpy.roll(data, 1, axis=1)

    # make three classes:
    print(len(numpy.where(data[:, 0] <= 2)[0]))
    print(len(numpy.where((data[:, 0] <= 5))[0]))
    print(len(numpy.where((data[:, 0] >= 6) & (data[:, 0] <= 7))[0]))
    print(len(numpy.where((data[:, 0] == 7))[0]))

    return "WineQuality", data, labels


def removeOutliers(dd, deviations=5):
    (dsname, data, featureNames) = dd
    print(list(range(data.shape[1])))
    for col in range(data.shape[1]):
        colsdata = data[:, col]
        dataidx = abs(colsdata - numpy.median(colsdata)) < deviations * numpy.std(colsdata)
        data = data[dataidx, ]

    return (dsname, data, featureNames)


DIST_TYPES = ['continuous', 'discrete', 'categorical']
RAND_SEED = 1337
FEATURE_FAMILIES = {'continuous': ['normal', 'beta', 'gamma', 'exponential', 'gumbel'],
                    'discrete': ['geometric', 'poisson', 'binomial'],
                    'categorical': ['bernoulli', 'categorical']}
FAMILY_PARAMETER_RANGES = {'normal': {'loc': [-100, 100], 'scale': [0, 3]},
                           'beta': {'a': [0, 20], 'b': [0, 20]},
                           'gamma': {'shape': [0, 20], 'scale': [0, 20]},
                           'exponential': {'scale': [0, 10]},
                           'gumbel': {'loc': [-10, 10], 'scale': [0, 3]},
                           'geometric': {'p': [0, 1]},
                           'poisson': {'lam': [0, 100]},
                           'binomial': {'n': [0, 100], 'p': [0, 1]},
                           'bernoulli': {'p': [0, 1]},
                           'categorical': {'k': [2, 100]}}


def sample_feature_param(distribution, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    params = None
    if distribution == 'binomial':
        params = {'n': rand_gen.choice(
            numpy.arange(*(FAMILY_PARAMETER_RANGES['binomial']['n'])).astype(int))}
        params['p'] = rand_gen.rand()
    elif distribution == 'poisson':
        params = {'lam': rand_gen.choice(
            numpy.arange(*(FAMILY_PARAMETER_RANGES['poisson']['lam'])).astype(int))}
    elif distribution == 'categorical':
        params = {'k': rand_gen.choice(
            numpy.arange(*(FAMILY_PARAMETER_RANGES['categorical']['k'])).astype(int))}
        params['p'] = rand_gen.rand(params['k'])
        params['p'] = params['p'] / sum(params['p'])
    else:
        #
        # sample uniformly
        params = {}
        for p_name, p_range in FAMILY_PARAMETER_RANGES[distribution].items():
            assert len(p_range) == 2
            params[p_name] = rand_gen.uniform(p_range[0], p_range[1])

    return params


def sample_distribution(distribution, params, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    if distribution == 'bernoulli':
        return rand_gen.binomial(n=1, p=params['p'])
    elif distribution == 'categorical':
        return rand_gen.choice(params['k'], p=params['p'])
    else:
        try:
            return getattr(rand_gen, distribution)(**params)
        except:
            raise ValueError('Unrecognized distribution {}'.format(distribution))


def generate_random_instance_indep(feature_families, params, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    n_features = len(feature_families)
    rand_instance = numpy.array([sample_distribution(feature_families[j],
                                                     params[j],
                                                     rand_gen)
                                 for j in range(n_features)])
    return rand_instance


def generate_random_instance_indep_mixture(feature_families,
                                           params,
                                           cluster_priors,
                                           rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    n_features = len(feature_families)
    assert len(cluster_priors) == n_features

    rand_instance = []
    instance_families = []
    for j in range(n_features):
        f = rand_gen.choice(numpy.arange(len(cluster_priors[j])),
                            p=cluster_priors[j])
        x_j = sample_distribution(feature_families[j][f],
                                  params[j][f],
                                  rand_gen)
        instance_families.append(feature_families[j][f])
        rand_instance.append(x_j)
    rand_instance = numpy.array(rand_instance)

    return rand_instance, instance_families


def generate_indep_synthetic_hybrid_random_dataset(n_instances,
                                                   n_features=10,
                                                   type_priors=None,
                                                   family_priors=None,
                                                   rand_gen=None,
                                                   dtype=numpy.float64):
    """
    Synthesize a dataset of M instances and N features such that each feature
    type is drawn from a prior distribution over types,
    then a family for that feature is drawn from a prior distribution
    according to the previously selected type.
    Each instance is sampled i.i.d from the joint probability distribution
    of all features considered independent one from the other
    """

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    #
    # generating priors
    n_types = len(DIST_TYPES)
    if type_priors is None:
        type_priors = numpy.ones(n_types) / n_types
        print('Uniform priors for types {}'.format(type_priors))

    if family_priors is None:
        family_priors = {type: (numpy.ones(len(families)) / len(families))
                         for type, families in FEATURE_FAMILIES.items()}
        print('Uniform priors for families {}'.format(family_priors))

    #
    # sampling a RV type
    feature_types = [rand_gen.choice(DIST_TYPES, p=type_priors)
                     for i in range(n_features)]

    #
    # each feature comes from a single RV family
    feature_families = [rand_gen.choice(FEATURE_FAMILIES[t], p=family_priors[t])
                        for t in feature_types]

    assert len(feature_types) == len(feature_families)
    print('Sampled feature types {}'.format(feature_types))
    print('Sampled feature families {}'.format(feature_families))

    #
    # sampling parameters for distributions
    feature_params = [sample_feature_param(f, rand_gen)
                      for f in feature_families]
    print('Sampled feature params {}'.format([(f, p)
                                              for f, p in zip(feature_families,
                                                              feature_params)]))

    rand_data = numpy.zeros((n_instances, n_features), dtype=dtype)
    for i in range(n_instances):
        rand_data[i] = generate_random_instance_indep(feature_families,
                                                      feature_params,
                                                      rand_gen)

    print(rand_data)
    return rand_data


def generate_indep_mixture_synthetic_hybrid_random_dataset(n_instances,
                                                           n_features=10,
                                                           n_clusters=None,
                                                           type_priors=None,
                                                           cluster_priors=None,
                                                           family_priors=None,
                                                           rand_gen=None,
                                                           dtype=numpy.float64):
    """
    Synthesize a dataset of M instances and N features such that each feature
    type is drawn from a prior distribution over types.
    Then for each feature one determines the number of different families (clusters) for that feature.
    After that, a family for each cluster, for a feature, is drawn from a prior distribution
    according to the previously selected type.
    Each instance is sampled i.i.d from the joint probability distribution
    of all features considered independent one from the other.
    """

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    #
    # assuming 2 clusters per features if not specified
    if n_clusters is None:
        n_clusters = numpy.zeros(n_features).astype(int)
        n_clusters[:] = 2
    print('Feature clusters {}'.format(n_clusters))

    #
    # generating priors
    n_types = len(DIST_TYPES)
    if type_priors is None:
        type_priors = numpy.ones(n_types) / n_types
        print('Uniform priors for types {}'.format(type_priors))

    if cluster_priors is None:
        cluster_priors = []
        for nc in n_clusters:
            c_prior = rand_gen.rand(nc)
            cluster_priors.append(c_prior / c_prior.sum())
        print('Uniform priors for feature clusters {}'.format(cluster_priors))

    if family_priors is None:
        family_priors = {type: (numpy.ones(len(families)) / len(families))
                         for type, families in FEATURE_FAMILIES.items()}
        print('Uniform priors for families {}'.format(family_priors))

    #
    # sampling cluster numbers
    # cluster_numbers = [rand_gen.choice(nc, p=nc_prior)
    #                    for nc, nc_prior in zip(n_clusters,
    #                                            cluster_priors)]

    #
    # sampling a RV type
    feature_types = [rand_gen.choice(DIST_TYPES, p=type_priors)
                     for i in range(n_features)]
    print('Sampled feature types {}'.format(feature_types))

    #
    # each feature can have different feature families
    feature_families = [rand_gen.choice(FEATURE_FAMILIES[t],
                                        p=family_priors[t],
                                        size=nc,
                                        replace=False)
                        for t, nc in zip(feature_types, n_clusters)]

    assert len(feature_types) == len(feature_families)
    print('Sampled feature families {}'.format(feature_families))

    #
    # sampling parameters for distributions
    feature_params = [[sample_feature_param(f, rand_gen) for f in ff]
                      for ff in feature_families]
    print('Sampled feature params {}'.format([(f, p)
                                              for f, p in zip(feature_families,
                                                              feature_params)]))

    rand_data = numpy.zeros((n_instances, n_features), dtype=dtype)
    for i in range(n_instances):
        rand_data[i], rand_families = generate_random_instance_indep_mixture(feature_families,
                                                                             feature_params,
                                                                             cluster_priors,
                                                                             rand_gen)
        print('{}\t{}'.format(i, '\t'.join(f for f in rand_families)))

    print(rand_data)
    return rand_data


def remove_missing_value_samples(data_path,
                                 missing_value_str='?,',
                                 header=1,
                                 output=None):
    """
    Load a UCI dataset as text lines from file,
    remove samples (lines) with missing values
    then return them and save those back to plain text optionally
    """

    lines = None
    processed_lines = None
    with open(data_path, 'r') as f:
        lines = f.readlines()
        logging.info('Loaded {} lines from {}'.format(len(lines),
                                                      data_path))

        #
        # removing empty lines
        lines = [l for l in lines if l.strip()]
        logging.info('{} samples after removing empty lines'.format(len(lines)))

        #
        # discarding the header?
        if header:
            lines = lines[header:]
            logging.info('After removing header there are {} samples'.format(len(lines)))

        processed_lines = [l for l in lines if missing_value_str not in l]
        logging.info('>>> {} lines without missing values'.format(len(processed_lines)))

    if output is not None:
        with open(output, 'w') as f:
            f.writelines(processed_lines)
            logging.info('Saved lines to {}!'.format(output))

    return processed_lines


def convert_value(value, feature_type, domain, float_dtype=numpy.float64):
    """
    domain is a map from strings to integers
    """
    if feature_type == "continuous":
        return float_dtype(value)

    elif feature_type == "discrete":
        #
        # FIXME this is ugly
        # if value == '--':
        #     return -1
        # else:
        return domain[value]

    elif feature_type == "categorical":
        return domain[value]


def load_uci_data_from_text(data_path,
                            sep=',',
                            invis='\n.',
                            header=1):
    """
    Loading data from a UCI file into a numpy array
    feature_domains is a list of maps (or None values)
    """

    lines = None
    with open(data_path, 'r') as f:
        lines = f.readlines()
    logging.info('Loaded {} lines from {}'.format(len(lines),
                                                  data_path))
    #
    # discarding the header?
    if header:
        lines = lines[header:]
        logging.info('After removing header there are {} samples'.format(len(lines)))

    #
    # tokenizing
    tokenized_data = [[t.strip().strip(invis) for t in l.split(sep)] for l in lines]

    return tokenized_data


def estimate_categorical_domains_old(tdatas,
                                     feature_types):

    # tdatas = [load_uci_data_from_text(d, sep=sep,
    #                                   invis=invis,
    #                                   header=header,
    #                                   dtype=dtype) for d in data_paths]

    cdomains = []
    for ft in feature_types:
        if ft == 'categorical':
            c = Counter()
            cdomains.append(c)
        else:
            cdomains.append(None)

    for td in tdatas:
        for sample in td:
            for j, ft in enumerate(feature_types):
                if ft == 'categorical':
                    cdomains[j][sample[j]] += 1

    return cdomains


def estimate_categorical_domain(tdata,
                                feature_id):

    # tdatas = [load_uci_data_from_text(d, sep=sep,
    #                                   invis=invis,
    #                                   header=header,
    #                                   dtype=dtype) for d in data_paths]

    cdomain = {}
    i = 0
    for sample in tdata:
        val = sample[feature_id]
        if val not in cdomain:
            cdomain[val] = i
            i += 1

    return cdomain


def estimate_continuous_domain_range(tdata,
                                     feature_id):
    """
    For a continuous feature we compute the min and max value
    to estimate its range
    """

    cdomain = {}
    cdomain['min'] = numpy.inf
    cdomain['max'] = -numpy.inf

    for sample in tdata:
        val = float(sample[feature_id])
        cdomain['min'] = min(val, cdomain['min'])
        cdomain['max'] = max(val, cdomain['max'])

    return cdomain


def estimate_continuous_domain(tdata,
                               feature_id,
                               binning_method=BINNING_METHOD,
                               float_type=numpy.float64,
                               range=None):
    """
    For a continuous feature we compute the numpy histogram
    """

    data = numpy.array([float(sample[feature_id]) for sample in tdata])

    if binning_method == 'unique':
        data = numpy.concatenate([data, numpy.array(range)])
        binning_method = numpy.unique(data)

    h, bins = compute_histogram(data,
                                bins=binning_method,
                                range=range)

    #print('BINZ', bins, 'h', h)
    # print('DOMAINZ', domains)

    # cdomain = {v: i for i, v in enumerate(domains)}
    cdomain = {v: i for i, v in enumerate(bins)}

    return cdomain


def estimate_continuous_domain_min_max(tdata,
                                       feature_id,
                                       binning_method='fd',
                                       float_type=numpy.float64,
                                       min=None,
                                       max=None):
    """
    For a continuous feature we compute the numpy histogram
    """

    data = numpy.array([float(sample[feature_id]) for sample in tdata] + [min, max])
    # data = numpy.array([float(sample[feature_id]) for sample in tdata])

    # # bins, domains = estimate_bins_and_domains_continuous_data(data,
    # #                                                           binning_method=[binning_method],
    # #                                                           float_type=float_type,
    # #                                                           range=range)

    # h, bins = compute_histogram(data,
    #                             bins=binning_method)

    # print('BINZ', bins)
    # # print('DOMAINZ', domains)

    bins = numpy.unique(data)

    #cdomain = {v: i for i, v in enumerate(domains)}
    cdomain = {v: i for i, v in enumerate(bins)}

    return cdomain


def uci_data_to_numpy(tdata,
                      feature_types,
                      feature_domains,
                      # sep=',',
                      # invis='\n.',
                      # header=1,
                      dtype=numpy.float64):

    print(feature_domains)
    assert len(feature_types) == len(feature_domains), (len(feature_types), len(feature_domains))

    # tdata = load_uci_data_from_text(data_path,
    #                                 sep=sep,
    #                                 invis=invis,
    #                                 header=header)

    #
    # based on feature_types, do a conversion
    n_features = len(feature_types)
    n_instances = len(tdata)
    # tdata = numpy.empty((n_instances, n_features), dtype=object)

    data = numpy.zeros((n_instances, n_features), dtype=dtype)
    data[:] = numpy.nan

    #
    # estimate collect domains for categorical data
    for i, sample in enumerate(tdata):
        assert len(sample) == n_features
        for j, (ft, fd) in enumerate(zip(feature_types, feature_domains)):
            data[i, j] = convert_value(sample[j], feature_type=ft, domain=fd, float_dtype=dtype)

    return data


FEATURE_TYPES = {'continuous', 'categorical', 'discrete'}


def load_feature_info_preprocess(feature_info_path,
                                 endline='.',
                                 sep=':',
                                 domain_sep=','):
    """
    Loads a .feature file and returns
    a list of feature names (strings)
    a list of feature types in FEATURE_TYPES
    a list of maps with possibly None values (for continuos features, the map
    has keys 'min' and 'max' and real values, for discrete and categorical features
    it is integer -> integer)
    """

    lines = None
    with open(feature_info_path, 'r') as f:
        lines = f.readlines()

    logging.info('Read {} lines from {}'.format(len(lines),
                                                feature_info_path))
    #
    # removing empty lines
    lines = [l for l in lines if l.strip()]
    logging.info('{} samples after removing empty lines'.format(len(lines)))

    feature_names = []
    feature_types = []
    domains = []

    # cont_domain_str = ['min', 'max']

    for l in lines:
        #
        # stripping endline
        l = l.strip()
        assert l[-1] == endline, l
        # tokenizing
        tokens = l[:-1].split(sep)

        assert len(tokens) == 2 or len(tokens) == 3, len(tokens)

        #
        # tokens[0] RV name
        # tokens[1] RV type
        # tokens[2] optional domain
        feature_names.append(tokens[0].strip())
        ftype = tokens[1].strip()
        assert ftype in FEATURE_TYPES, ftype
        feature_types.append(ftype)
        if len(tokens) == 3:
            domain_str = tokens[2].strip()

            #
            # categorical case, map everything to integers from 0 to K
            if ftype == 'categorical':
                domains.append({d.strip(): i for i, d in enumerate(domain_str.split(domain_sep))})
            #
            # discrete, map everything that is not a number to an unused number
            elif ftype == 'discrete':
                disc_domain = {}
                used_ints = set()
                for i, d in enumerate(domain_str.split(domain_sep)):
                    d_i = None
                    try:
                        d_i = int(d)
                        assert d_i not in used_ints

                    except:
                        if not used_ints:
                            d_i = 0
                        else:
                            d_i = max(used_ints) + 1

                    assert d_i is not None
                    disc_domain[d.strip()] = d_i
                    used_ints.add(d_i)

                domains.append(disc_domain)
            #
            # min and max ranges
            elif ftype == 'continuous':
                # disc_domain = {m: d for m, d in zip(cont_domain_str, domain_str.split(domain_sep))}
                disc_domain = {float(m): _i for _i, m in enumerate(domain_str.split(domain_sep))}
                print('CD', disc_domain)
                domains.append(disc_domain)

            else:
                raise NotImplementedError('Unrecognized feature type {}'.format(tokens))
        else:
            domains.append(None)

    return feature_names, feature_types, domains


def load_feature_info(feature_info_path,
                      endline='.',
                      sep=':',
                      domain_sep=',',
                      float_dtype=numpy.float64,
                      int_dtype=numpy.int32):
    """
    Loads a .feature file and returns
    a list of feature names (strings)
    a list of feature types in FEATURE_TYPES
    a list of ordered numpy arrays of floats (continuous features) or integers (discrete and categorical))
    """

    lines = None
    with open(feature_info_path, 'r') as f:
        lines = f.readlines()

    logging.info('Read {} lines from {}'.format(len(lines),
                                                feature_info_path))
    #
    # removing empty lines
    lines = [l for l in lines if l.strip()]
    logging.info('{} samples after removing empty lines'.format(len(lines)))

    feature_names = []
    feature_types = []
    domains = []

    # cont_domain_str = ['min', 'max']

    for l in lines:
        #
        # stripping endline
        l = l.strip()
        assert l[-1] == endline, l
        # tokenizing
        tokens = l[:-1].split(sep)

        assert len(tokens) == 2 or len(tokens) == 3, len(tokens)

        #
        # tokens[0] RV name
        # tokens[1] RV type
        # tokens[2] optional domain
        feature_names.append(tokens[0].strip())
        ftype = tokens[1].strip()
        assert ftype in FEATURE_TYPES, ftype
        feature_types.append(ftype)
        if len(tokens) == 3:
            domain_str = tokens[2].strip()
            domain_vals = numpy.array([d.strip() for d in domain_str.split(domain_sep)])

            #
            # categorical and discrete case, map everything to integers from 0 to K
            if ftype == 'categorical' or ftype == 'discrete':
                domain_vals = domain_vals.astype(int_dtype)

            #
            # discrete, map everything that is not a number to an unused number
            elif ftype == 'continuous':
                domain_vals = domain_vals.astype(float_dtype)

            else:
                raise NotImplementedError('Unrecognized feature type {}'.format(tokens))
            domains.append(numpy.sort(domain_vals))
        else:
            domains.append(None)

    return feature_names, feature_types, domains


def save_feature_info_dict(feature_names, feature_types, feature_domains, out_path, domain_keys=True, range=True):

    lines = []
    for fn, ft, fd in zip(feature_names, feature_types, feature_domains):

        domain_str = None

        if fd is not None:
            #
            # FIXME: should be more appropriate to have it the other way around...
            # key= integer position value=actual value
            sorted_domain = sorted(fd.items(), key=operator.itemgetter(1))
            if ft == 'continuous':
                if range:
                    domain_str = ':{},{}'.format(fd['min'], fd['max'])
                else:
                    domain_str = ':{}'.format(','.join(str(s) for s, _ in sorted_domain))

            elif ft == 'categorical' or ft == 'discrete':
                if domain_keys:
                    domain_str = ':{}'.format(','.join(str(d) for d, _ in sorted_domain))
                else:
                    domain_str = ':{}'.format(','.join(str(d) for _, d in sorted_domain))
            else:
                raise ValueError('Unrecognized feature type')
        else:
            domain_str = ''

        line = '{}:{}{}.\n'.format(fn, ft, domain_str)
        lines.append(line)

    with open(out_path, 'w') as f:
        f.writelines(lines)
        logging.info('Wrote feature info to {}'.format(out_path))

    return lines


def save_feature_info(feature_names, feature_types, feature_domains, out_path):

    lines = []
    for fn, ft, fd in zip(feature_names, feature_types, feature_domains):

        domain_str = ','.join(str(d) for d in fd)

        line = '{}:{}:{}.\n'.format(fn, ft, domain_str)
        lines.append(line)

    with open(out_path, 'w') as f:
        f.writelines(lines)
        logging.info('Wrote feature info to {}'.format(out_path))

    return lines


def get_feature_formats(feature_types):
    formats = []
    for f in feature_types:
        if f in {'discrete', 'categorical'}:
            formats.append('%d')
        elif f == 'continuous':
            formats.append('%.5f')
        else:
            raise ValueError('Unreconized feature value {}'.format(f))

    return formats


def loadMLC(dsname, base_path=os.path.dirname(__file__), data_dir="datasets/MLC/proc-db/proc/"):

    dataset_path = os.path.join(base_path, data_dir, dsname)
    feature_names, feature_types, domains = load_feature_info('{}.features'.format(dataset_path))

    train = numpy.loadtxt("{}.train.data".format(dataset_path), delimiter=",")
    test = numpy.loadtxt("{}.test.data".format(dataset_path), delimiter=",")
    valid = numpy.loadtxt("{}.valid.data".format(dataset_path), delimiter=",")

    return ((train, valid, test), feature_names, feature_types, domains)


if __name__ == '__main__':
    print(getCommunitiesAndCrimes()[1].shape)

    print(removeOutliers(getCommunitiesAndCrimes())[1].shape)
    # print(len(getNips()[2]))
    # print(getCommunitiesAndCrimes())
    # print(getSynthetic())
    # print(getMSNBCclicks())
    # print(getGrolier())
