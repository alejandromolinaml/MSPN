from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import sys
from math import log
from math import exp

import numpy
import astropy.stats

from tfspn.histogram import getHistogramVals

# from rpy2 import robjects
# from rpy2.robjects.packages import importr
# from rpy2.robjects import numpy2ri
# from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

# marginalize indicator
MARG_IND = -1

# log of zero const, to avoid -inf
# numpy.exp(LOG_ZERO) = 0
LOG_ZERO = -1e3
RAND_SEED = 1337


def IS_LOG_ZERO(log_val):
    """
    checks for a value to represent the logarithm of 0.
    The identity to be verified is that:
    IS_LOG_ZERO(x) && exp(x) == 0
    according to the constant LOG_ZERO
    """
    return (log_val <= LOG_ZERO)


# defining a numerical correction for 0
EPSILON = sys.float_info.min

# size for integers
INT_TYPE = 'int8'

# seed for random generators
RND_SEED = 31

# negative infinity for worst log-likelihood
NEG_INF = -sys.float_info.max

# EPS = numpy.finfo(numpy.float).eps

NODE_SYM = 'u'  # unknown type
SUM_NODE_SYM = '+'
PROD_NODE_SYM = '*'
INDICATOR_NODE_SYM = 'i'
DISCRETE_VAR_NODE_SYM = 'd'
CHOW_LIU_TREE_NODE_SYM = 'c'
CONSTANT_NODE_SYM = 'k'


BINNING_METHOD = "auto"


# class Node(object):

#     """
#     WRITEME
#     """
#     # class id counter
#     id_counter = 0

#     def __init__(self, var_scope=None):
#         """
#         WRITEME
#         """
#         # default val is 0.
#         self.log_val = LOG_ZERO

#         # setting id and incrementing
#         self.id = Node.id_counter
#         Node.id_counter += 1

#         # derivative computation
#         self.log_der = LOG_ZERO

#         self.var_scope = var_scope

#         # self.children_log_vals = numpy.array([])

#     def __repr__(self):
#         return 'id: {id} scope: {scope}'.format(id=self.id,
#                                                 scope=self.var_scope)

#     # this is probably useless, using it for test purposes
#     def set_val(self, val):
#         """
#         WRITEME
#         """
#         if numpy.allclose(val, 0, 1e-10):
#             self.log_val = LOG_ZERO
#         else:
#             self.log_val = log(val)

#     def __hash__(self):
#         """
#         A node has a unique id
#         """
#         return hash(self.id)

#     def __eq__(self, other):
#         """
#         WRITEME
#         """
#         return self.id == other.id

#     def node_type_str(self):
#         return NODE_SYM

#     def node_short_str(self):
#         return "{0} {1}\n".format(self.node_type_str(),
#                                   self.id)

#     @classmethod
#     def reset_id_counter(cls):
#         """
#         WRITEME
#         """
#         Node.id_counter = 0

#     @classmethod
#     def set_id_counter(cls, val):
#         """
#         WRITEME
#         """
#         Node.id_counter = val


def is_piecewice_linear_pdf(x, y):
    return numpy.allclose(numpy.trapz(y, x), 1.0)


def isotonic_unimodal_regression_R(x, y):
    """
    Perform unimodal isotonic regression via the Iso package in R
    """

    numpy2ri.activate()
    # n_instances = x.shape[0]
    # assert y.shape[0] == n_instances

    importr('Iso')
    z = robjects.r["ufit"](y, x=x, type='b')
    iso_x, iso_y = numpy.array(z.rx2('x')), numpy.array(z.rx2('y'))

    return iso_x, iso_y

import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def continuous_data_to_piecewise_linear_pdf(data,
                                            disc_method="histogram",
                                            n_bins=BINNING_METHOD,
                                            interval=None,
                                            alpha=None,
                                            return_histogram=False):
    """
    First we have to discretize the data,
    then we 'fit' a piecewise linear function to represent the pdf
    Outputs two sequences X = x1, x2, ..., xn ; Y = y1, y2, ..., yn representing
    the piecewise linear
    """

    n_instances = data.shape[0]
    disc_data, bins, = None, None
    if disc_method == 'histogram':
        #
        # discretization through an histogram
        disc_data, bins = numpy.histogram(data, bins=n_bins, range=interval, density=False)
        assert len(disc_data) + 1 == len(bins)
    else:
        raise ValueError('{} discretization method not implemented'.format(disc_method))

    n_vals = len(bins) - 1
    #
    # apply Laplace smoothing to the histogram
    if alpha is not None:
        disc_data = (disc_data + alpha) / (n_instances + n_vals * alpha)
    else:
        disc_data = disc_data / n_instances

    # #
    # # histogram smoothing by mean shift?
    # for i in range(1, len(bins) - 1):
    #     disc_data[i] = numpy.sum(disc_data[i - 1:i + 2]) / 3

    #
    # getting the line through the points centered in the bins
    x, y = [], []
    for (b0, b1), f in zip(pairwise(bins), disc_data):
        x.append(b0 + (b1 - b0) / 2)
        y.append(f)

    if return_histogram:
        return numpy.array(x), numpy.array(y), (disc_data, bins)
    else:
        return numpy.array(x), numpy.array(y)


# def make_piecewise_linear_node(data, var,
#                                disc_method="histogram",
#                                n_bins=BINNING_METHOD,
#                                interval=None,
#                                alpha=None,
#                                isotonic=True,
#                                normalize=True):

#     #
#     # discretize in some way
#     x, y = continuous_data_to_piecewise_linear_pdf(data,
#                                                    disc_method=disc_method,
#                                                    n_bins=n_bins,
#                                                    interval=interval,
#                                                    alpha=alpha)

#     #
#     # going isotonic unimodal on the discretizations?
#     if isotonic:
#         x, y = isotonic_unimodal_regression_R(x, y, normalize=normalize)

#     #
#     # use the piecewise line to now create a node
#     leaf_node = PiecewiseLinearPDFNode(var, x, y)

#     return leaf_node


def domain_to_bins(domain,
                   step=1,
                   center=False):
    """
    Get a bin partitioning as a sequence X = (x0, x1, ..., xn+1)
    from a domain represented as an ordered sequence of points D = (d0, d1, ..., dn)
    step: the size of each bin (defaults to 1)
    if center is True:
      D will be the sequence of the centers of the bins in X, e.g. d0 = x0 + step/2 = x1 - step/2
    otherwise:
      D will be the sequence of the starting elements in X, e.g. d0 = x0, ..., dn = xn, xn+1 = dn+ step
    """
    #
    # filling missing gaps
    domain = numpy.arange(domain.min(), domain.max() + step, step).astype(domain.dtype)
    bins = None
    if center:
        bins = numpy.append(domain - step / 2, domain[-1] + step / 2)
    else:
        bins = numpy.append(domain, domain[-1] + step)

    assert len(bins) == len(domain) + 1, (len(bins), len(domain) + 1)

    return bins


def compute_histogram(data,
                      bins=BINNING_METHOD,
                      range=None,
                      density=False):
    """
    Just a wrapper around numpy.histogram
    """
    #
    # use astropy
    if bins == 'blocks':
        #
        # including min and max ranges
        data = numpy.concatenate([data, numpy.array(range)])
        # print('dataz', data)

        h, b = astropy.stats.histogram(data, bins="blocks", range=range)

        if len(h) > 0:
            return h, b
        else:
            print('reverting to numpy auto histogram')
            bins = "auto"
    #
    # using numpy
    # else:
    return numpy.histogram(data, bins=bins, range=range, density=density)


def compute_histogram_type_wise(data,
                                feature_type,
                                domain,
                                alpha=0.0):


    repr_points = None

    if feature_type == 'continuous':
        maxx = numpy.max(domain)
        minx = numpy.min(domain)
        
        prior_density = 1 / (maxx - minx)

        if numpy.var(data) > 1e-10:
            breaks, densities, mids = getHistogramVals(data)
        else:
            breaks = numpy.array([minx, maxx])
            densities = numpy.array([prior_density])
            mids = numpy.array([minx + (maxx - minx) / 2])

        repr_points = mids

    elif feature_type in {'discrete', 'categorical'}:
        prior_density = 1 / len(domain)

        #
        # augmenting one fake bin left
        breaks = numpy.array([d for d in domain] + [domain[-1] + 1])
        # print('categorical binning', breaks)
        # if numpy.var(data_slice.getData()) == 0.0:
        densities, breaks = compute_histogram(data,  bins=breaks, density=True)

        repr_points = domain
        #
        # laplacian smoothing?
        if alpha:
            n_samples = data.shape[0]
            n_bins = len(breaks) - 1
            counts = densities * n_samples
            densities = (counts + alpha) / (n_samples + n_bins * alpha)

    assert len(densities) == len(repr_points)
    assert len(densities) == len(breaks) - 1

    return densities, breaks, prior_density, repr_points

# def compute_histogram2(data,
#                        bins=BINNING_METHOD,
#                        range=None):
#     """
#     Just a wrapper around numpy.histogram
#     """
#     # return astropy.stats.histogram(data,bins="blocks",range=range)
#     return numpy.histogram(data, bins=bins, range=range, density=False)


def averaging_histograms(histograms):
    """
    Average over a sequence of histograms (pairs of (counts, bins)) sharing the same bins
    """
    for counts, hbins in histograms:
        assert len(hbins) == len(counts) + 1, (len(hbins), len(counts) + 1)
        assert numpy.allclose(hbins, histograms[0][1])

    counts = numpy.mean([d for d, _b in histograms], axis=0)
    hbins = histograms[0][1]
    assert len(counts) + 1 == len(hbins), (len(hbins), len(counts) + 1)

    return counts, hbins


def histogram_counts_to_frequences(histogram, alpha=None):
    """
    Processing histogram counts into frequences, possibly
    applying laplace smoothing
    """
    counts, bins = histogram
    n_instances = numpy.sum(counts)

    #
    # laplace smoothing?
    if alpha is not None:
        n_vals = len(bins) - 1
        freqs = (counts + alpha) / (n_instances + n_vals * alpha)
    else:
        freqs = counts / n_instances

    return freqs, bins


def piecewise_linear_from_histogram_old(disc_data, bins, center=False, step=1):
    """
    Extract a piecewise linear representation from an histogram representations
    where bins is an ordered sequence of scalars (bin extrema) and disc_data is
    the bin frequences
    If center: center of the bin intervals are considered, otherwise, the first bin extremum (left point)
    """
    x, y = None, None
    if center:
        # print('centering')
        x = [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(bins)]
        x = [x[0] - step] + x + [x[-1] + step]
        # print('new x', x)
        y = [0.0] + [d for d in disc_data] + [0.0]
    else:
        # x = [b for b in bins[:-1]]
        x = [b for b in bins]
        # x = [b for b in bins[1:]]
        # x = [x[0] - step] + x + [x[-1] + step]
        x = [x[0] - step] + x
        y = [0.0] + [d for d in disc_data] + [0.0]
        # print('no centering', x, y)

    assert len(x) == len(y), (len(x), len(y))
    assert len(x) == len(disc_data) + 2

    return numpy.array(x), numpy.array(y)


def piecewise_linear_from_histogram(disc_data, bins, center=False, step=1):
    """
    Extract a piecewise linear representation from an histogram representations
    where bins is an ordered sequence of scalars (bin extrema) and disc_data is
    the bin frequences

    If center: center of the bin intervals are considered, otherwise, the first bin extremum (left point)

    We now already assume that we have the right y
    """
    x, y = None, None
    if center:
        # print('centering')
        # print(bins)
        x = [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(bins)]
        # print(x)
        x = [x[0] - step] + x + [x[-1] + step]
        # print('new x', x)
        # y = [0.0] + [d for d in disc_data] + [0.0]
        # y = disc_data
        y = [0.0] + [d for d in disc_data] + [0.0]
    else:
        # x = [b for b in bins[:-1]]
        x = [b for b in bins]
        # x = [b for b in bins[1:]]
        # x = [x[0] - step] + x + [x[-1] + step]
        # x = [x[0] - step] + x
        # y = [0.0] + [d for d in disc_data] + [0.0]
        y = [d for d in disc_data] + [0.0]
        # print('no centering', x, y)

    assert len(x) == len(y), (len(x), len(y))
    # assert len(x) == len(disc_data) + 2

    return numpy.array(x), numpy.array(y)


EPS = 1e-8  # numpy.finfo(float).eps


def histogram_to_piecewise_linear_type_wise(densities, bins, feature_type):
    """
    De
    """

    assert len(densities) == len(bins) - 1
    x, y = None, None
    if feature_type == 'continuous':
        #
        # we use the bins centers and that's it
        # x = [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(bins)]
        # y = densities

        if len(densities) > 1:
            # x = [bins[0] - (bins[1] - bins[0]) / 2] + \
            #     [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(bins)] + \
            #     [bins[-1] + (bins[-1] - bins[-2]) / 2]
            # y = [0.0] + [d for d in densities] + [0.0]
            # x = [bins[0] - EPS] + \
            #     [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(bins)] + \
            #     [bins[-1] + EPS]

            # y = [densities[0]] + [d for d in densities] + [densities[-1]]
            x = [bins[0] - EPS] + \
                [b0 + (b1 - b0) / 2 for (b0, b1) in pairwise(bins)] + \
                [bins[-1] + EPS]

            y = [0.0] + [d for d in densities] + [0.0]
            # print('histo')

        else:
            assert len(bins) == 2
            x = [bins[0], bins[1]]
            y = [densities[0], densities[0]]
        #
        # dealing with single bins, triangle?
        # if len(x) == 1:
        #     x = [bins[0]] + x + [bins[1]]
        #     y = [0.0] + [densities[0]] + [0.0]

        #
        # piecewise constant
        if len(x) == 1:
            assert len(bins) == 2
            x = [bins[0], bins[1]]
            y = [densities[0], densities[0]]

    elif feature_type in {'discrete', 'categorical'}:

        #
        # we just augment the bins by adding the two 'tails
        tail_width = 1
        x = [b for b in bins[:-1]]
        x = [x[0] - tail_width] + x + [x[-1] + tail_width]

        y = [0.0] + [d for d in densities] + [0.0]

    else:
        raise ValueError('Unrecognized feature type {}'.format(feature_type))

    assert len(x) == len(y), (len(x), len(y))

    # print('before normalizing', x, y)
    #
    # normalizing
    x, y = numpy.array(x), numpy.array(y)
    auc = numpy.trapz(y, x)
    y = y / auc

    # print('after before normalizing', y, auc)

    return x, y


def piecewise_linear_to_unimodal_isotonic(x, y):
    """
    Applying unimodal isotonic regression to 
    """

    x, y = isotonic_unimodal_regression_R(x, y)

    auc = numpy.trapz(y, x)
    y = y / auc

    return x, y


def merge_piecewise_linear_observations(x_series, y_series, remove_duplicates=True):
    """
    Given two (multi-)sequences of ordered epoint sequences X = {{x_0^0, ..., x_n^0}, ..., {x_0^k, ..., x_n^k}},
    and Y = {{y_0^0, ..., y_n^0}, ..., {y_0^k, ..., y_n^k}}
    return two ordered sequences of points by unnesting X and Y
    """

    assert len(x_series) == len(y_series), (len(x_series), len(y_series))
    for x, y in zip(x_series, y_series):
        assert len(x) == len(y), (len(x), len(y))

    #
    # unnesting them
    u_x = numpy.array([x for sx in x_series for x in sx])
    u_y = numpy.array([y for sy in y_series for y in sy])
    #
    # ordering them back
    ord_x_ids = numpy.argsort(u_x)

    x = u_x[ord_x_ids]
    y = u_y[ord_x_ids]

    #
    # removing points with the same x?
    # the one to be removed is arbitrarly chosen by numpy btw
    if remove_duplicates:
        x, unique_x_ids = numpy.unique(x, return_index=True)
        y = y[unique_x_ids]

    return x, y


def equal_binning(bins_1, bins_2):
    if type(bins_1) != type(bins_2):
        return False
    elif isinstance(bins_1, str) or isinstance(bins_2, int):
        return bins_1 == bins_2
    else:
        return numpy.allclose(bins_1, bins_2)


def piecewise_linear_approximation(data,
                                   bins,
                                   family,
                                   bin_width=1,
                                   alpha=None,
                                   n_bootstraps=None,
                                   average_bootstraps=False,
                                   remove_duplicates=True,
                                   isotonic=True,
                                   rand_gen=None):
    """
    bins has to be a list of precomputed-binnings
    """

    # FIXME: this shall be done in a caller method
    # # getting bins from domain values
    # bins = domain_to_bins(domain,
    #                       step=bin_width,
    #                       center=center)

    n_instances = data.shape[0]
    samples = None

    center = None
    if family == 'continuous':
        center = False
        tail_width = bin_width
    elif family == 'discrete' or family == 'categorical':
        center = True
        tail_width = 1

    #
    # bootstrapping? (expecting a list of bins)
    if n_bootstraps is not None:
        assert len(bins) == n_bootstraps, (len(bins), n_bootstraps)
        # if isinstance(bins, str):
        #     raise ValueError('Cannot perform bootstrapping on histograms'
        #                      ' with potentially different bins')
        if rand_gen is None:
            rand_gen = numpy.random.RandomState(RND_SEED)

        bootstrap_indices = [rand_gen.choice(n_instances,
                                             size=n_instances,
                                             replace=True)
                             for k in range(n_bootstraps)]
        samples = [data[bootstrap_indices[k]]
                   for k in range(n_bootstraps)]

    elif len(bins) > 1 and not isinstance(bins, numpy.ndarray):
        samples = [data for i in range(len(bins))]

    else:
        samples = [data]

    assert len(bins) == len(samples), (len(bins), len(samples))

    #
    # computing histograms
    histograms = [compute_histogram(s, bins=b, range=None)
                  for s, b in zip(samples, bins)]
    # histograms = [(numpy.digitize(s, b), b) for s, b in zip(samples, bins)]
    # print(len(histograms[0][0]), len(histograms[0][1]), histograms[0][1].max())

    # print("histograms", histograms)

    #
    # averaging histograms? (expecting all bins in the list to be equal)
    if average_bootstraps:
        # print('averaging bootstraps')
        for bins_1, bins_2 in pairwise(bins):
            assert equal_binning(bins_1, bins_2)
        histograms = [averaging_histograms(histograms)]

    #
    # from counts to frequencies (and smoothing?)
    histograms = [histogram_counts_to_frequences(h, alpha)
                  for h in histograms]
    # print(len(histograms[0][0]), len(histograms[0][1]), histograms[0][1].max())

    #
    # piecewise linear functions from histogram tips
    piecewises = [piecewise_linear_from_histogram(f, b, center=center, step=tail_width)
                  for f, b in histograms]

    # print("histograms", histograms)
    # print("piecewises", piecewises)

    #
    # merging piecewise approximations
    x_series = []
    y_series = []
    for x, y in piecewises:
        x_series.append(x)
        y_series.append(y)
    x, y = merge_piecewise_linear_observations(x_series, y_series,
                                               remove_duplicates=remove_duplicates)

    #
    # isotonic?
    if isotonic:
        x, y = isotonic_unimodal_regression_R(x, y)

    auc = numpy.trapz(y, x)
    y = y / auc
    assert is_piecewice_linear_pdf(x, y), numpy.trapz(y, x)

    return x, y


def estimate_bins_and_domains_continuous_data(data,
                                              binning_method=[BINNING_METHOD],
                                              float_type=numpy.float64,
                                              range=None):
    """
    Return a sequence of N multi-binning and domains
    for a data column (data) of size (M,) or (M, 1)
    typed as 'continuous' data

    A multi-binning a sequence of binnings, default case: one binning, more binnings
    if one specifies different binning-methods.
    A domain is a sequence of x values

    For a 'continuous' feature get a multi-binning by computing more than one histogram.
    The domain is the range of ordered observations

    data: a numpy array (of floats?) of size (M,) or (M, 1)
    binning_method: a single histogram type or a sequence (see numpy's histogram)
    float_type: float type representation for bins and domains
    """

    assert data.ndim == 1 or (data.ndim == 2 and
                              data.shape[1] == 1), \
        "Expecting one column only {}".format(data.shape)

    domain = numpy.unique(data.astype(float_type))
    #
    # do we want to provide a larger interval?
    # see scipy's histogram
    # NOTE: numpy's histogram is flattening the array
    binning = []
    for b in binning_method:
        # print(b)
        h, bins = compute_histogram(data,
                                    bins=b,
                                    range=range)
        # print('BINS', bins)
        binning.append(bins)

    return binning, domain


def estimate_bins_and_domains_discrete_data(data,
                                            center_bins=True,
                                            int_type=numpy.int32):
    """
    Return a sequence of N multi-binning and domains
    for a data column (data) of size (M,) or (M, 1)
    typed as 'discrete' data

    For a 'discrete' feature its domain is its ordered observations
    Its multi-binning is a single-binning that has bins centered on the domain elements

    data: a numpy array (of floats?) of size (M,) or (M, 1)
    binning_method: a single histogram type or a sequence (see numpy's histogram)
    float_type: float type representation for bins and domains
    """

    assert data.ndim == 1 or (data.ndim == 2 and
                              data.shape[1] == 1), \
        "Expecting one column only {}".format(data.shape)

    domain = numpy.sort(data.astype(int_type))
    #
    # NOTE: having one-sized bins for normalization issues
    binning = [domain_to_bins(domain, step=1, center=center_bins)]

    return binning, domain


def estimate_bins_and_domains_categorical_data(data,
                                               center_bins=True,
                                               int_type=numpy.int32):
    """
    Return a sequence of N multi-binning and domains
    for a data column (data) of size (M,) or (M, 1)
    typed as 'discrete' data

    For a 'categorical' feature its domain is the set of unique values in the data
    (assumed to be already, arbitrarly, transformed to integer indexes  0 -> k) 
    Its multi-binning is a single-binning that has bins centered on the domain elements

    data: a numpy array (of floats?) of size (M,) or (M, 1)
    binning_method: a single histogram type or a sequence (see numpy's histogram)
    float_type: float type representation for bins and domains
    """

    assert data.ndim == 1 or (data.ndim == 2 and
                              data.shape[1] == 1), \
        "Expecting one column only {}".format(data.shape)

    domain = numpy.unique(data.astype(int_type))
    #
    # NOTE: having one-sized bins for normalization issues
    binning = [domain_to_bins(domain, step=1, center=center_bins)]

    return binning, domain


def estimate_bins_and_domains_from_data(data,
                                        type_labels,
                                        binning_method=[BINNING_METHOD],
                                        center_bins=True,
                                        float_type=numpy.float64,
                                        int_type=numpy.int32):
    """
    Return a sequence of N multi-binning and domains
    for a data matrix (data) of size (MxN)
    having weak annotations about the N features (type_labels) about it being
    'continuous'|'discrete'|'categorical'

    A multi-binning a sequence of binnings, default case: one binning, more binnings
    if one specifies different binning-methods.
    A domain is a sequence of values

    For a 'continuous' feature get a multi-binning by computing more than one histogram.
    The domain is the range of ordered observations

    For a 'discrete' feature its domain is its ordered observations
    Its multi-binning is a single-binning that has bins centered on the domain elements

    For a 'categorical' feature its domain is the set of unique values in the data
    (assumed to be already, arbitrarly, transformed to integer indexes  0 -> k) 
    Its multi-binning is a single-binning that has bins centered on the domain elements

    data: a numpy array (of floats?) of size M instances x N features
    type_labels: a sequence of N strings in {'continuous'|'discrete'|'categorical'}
    binning_method: a single histogram type or a sequence (see numpy's histogram)
    center_bins: wheter to center bins around domain values for discrete and categorical data
    float_type: float type representation
    int_type: int type representation
    """

    binnings, domains = [], []

    n_instances = data.shape[0]
    n_features = data.shape[1]
    assert len(type_labels) == n_features, (len(type_labels) == n_features)

    for f in range(n_features):

        domain, binning = None, None

        feature_data = data[:, f]
        feature_type = type_labels[f]

        if feature_type == 'continuous':
            binning, domain = estimate_bins_and_domains_continuous_data(feature_data,
                                                                        binning_method=binning_method,
                                                                        float_type=float_type)

        elif feature_type == 'discrete':
            binning, domain = estimate_bins_and_domains_discrete_data(feature_data,
                                                                      center_bins=center_bins,
                                                                      int_type=int_type)

        elif feature_type == 'categorical':
            binning, domain = estimate_bins_and_domains_categorical_data(feature_data,
                                                                         center_bins=center_bins,
                                                                         int_type=int_type)

        else:
            raise ValueError('Unrecognized feature type label : {}'.format(feature_type))

        binnings.append(binning)
        domains.append(domain)

    return binnings, domains


def estimate_domains(data, families, float_type=numpy.float64, int_type=numpy.int32):
    domains = []
    for i, family in enumerate(families):

        if family == 'continuous':
            col = data[:, i].astype(float_type)
        elif family == 'discrete':
            col = data[:, i].astype(int_type)
        elif family == 'categorical':
            col = data[:, i].astype(int_type)
        else:
            raise ValueError('Unrecognized feature type label : {}'.format(family))

        domains.append(numpy.unique(col))

    return domains


def estimate_domains_range(data, families, float_type=numpy.float64, int_type=numpy.int32):
    domains = []
    for i, family in enumerate(families):

        if family == 'continuous':
            col = data[:, i].astype(float_type)
            domains.append(numpy.array([col.min(), col.max()]))
        elif family == 'discrete':
            col = data[:, i].astype(int_type)
            domains.append(numpy.unique(col))
        elif family == 'categorical':
            col = data[:, i].astype(int_type)
            domains.append(numpy.unique(col))
        else:
            raise ValueError('Unrecognized feature type label : {}'.format(family))

    return domains


def estimate_bins_old(data, family, domain, binning_method=[BINNING_METHOD], center_bins=True, bin_width=1):
    """
    Return a sequence of N multi-binning 
    """

    if family == 'continuous':
        binning = []
        for b in binning_method:
            _h, bins = compute_histogram(data, bins=b, range=None)

            #
            # adding a fake last bin
            bins = numpy.array([b_i for b_i in bins] + [bins[-1] + bin_width])
            # print("aaaa", bins, b, domain)
            binning.append(bins)
        # print("bbbbbbbbbbbbbb", binning)
        return binning

    if family == 'discrete' or family == 'categorical':
        return [domain_to_bins(domain, step=1, center=center_bins)]

    raise ValueError('Unrecognized feature type label : {}'.format(family))


def estimate_bins(data, family, domains, binning_method=[BINNING_METHOD], center_bins=True, bin_width=1):
    """
    NOTE: domains is now a list of domains
    Return a sequence of N multi-binning 
    """

    if family == 'continuous':
        binning = []
        # for b in binning_method:
        #     _h, bins = compute_histogram(data, bins=b, range=None)
        #     # print(_h)
        #     #
        #     # adding a fake last bin
        #     bins = numpy.array([b_i for b_i in bins] + [bins[-1] + bin_width])
        #     binning.append(bins)
        # return binning

        for d in domains:
            # print('\n\n\n\n\domain {}\n\n\n\n'.format(d))
            b = numpy.array([d[0] - bin_width] + [d_i for d_i in d] + [d[-1] + bin_width])
            _h, bins = compute_histogram(data, bins=b, range=None)
            # print(_h)
            #
            # adding a fake last bin

            binning.append(bins)
        return binning

    if family == 'discrete' or family == 'categorical':
        assert len(domains) == 1
        return [domain_to_bins(domains[0], step=1, center=center_bins)]

    raise ValueError('Unrecognized feature type label : {}'.format(family))

#
# SAMPLING UTILS
#


def compute_bin_masses(x, y, normalize=True):
    """
    Determine the (possibly unnormalized) mass falling inside each bin,
    highlighted by the piecewise linear function and
    represented by the two sequences of points, x and y
    NOTE: if the piecewise linear is a valid density,
    it shall be already normalized
    """

    assert len(x) == len(y)

    masses = []

    for (x_1, y_1), (x_2, y_2) in pairwise(zip(x, y)):
        masses.append(numpy.trapz([y_1, y_2], [x_1, x_2]))

    masses = numpy.array(masses)
    if normalize:
        masses = masses / masses.sum()
    assert len(masses) == len(x) - 1

    return masses


def rejection_sampling_from_trapezoid(x, y, n_samples=1, rand_gen=None):
    """
    The most basic way to sample from a trapezoid
    NOTE: if len(x) = len(y) > 2, then we have a piecewise linear density
    => the longer the sequence, the more inefficient the sampling
    """

    assert len(x) == len(y)

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    #
    # determine the rectangle including the trapezoid
    box_w_0 = min(x)
    box_w_1 = max(x)
    box_h = max(y)

    samples = []
    while len(samples) < n_samples:
        #
        # generate random point uniformly in the box
        r_x = rand_gen.uniform(box_w_0, box_w_1)
        r_y = rand_gen.uniform(0, box_h)
        #
        # is it in the trapezoid?
        trap_y = numpy.interp(r_x, xp=x, fp=y)
        if r_y < trap_y:
            samples.append(r_x)

    return numpy.array(samples)


def two_staged_sampling_piecewise_linear(x, y, masses=None, n_samples=1, sampling='rejection', rand_gen=None):
    """
    FIrst randomly select a binned trapezoid proportional to its mass,
    Then sample from it (rejection sampling implemented now)
    """

    assert len(x) == len(y)

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    if masses is None:
        masses = compute_bin_masses(x, y, normalize=True)

    n_bins = len(masses)
    samples = []
    while len(samples) < n_samples:
        #
        # randomly get a trapezoid
        rand_bin = rand_gen.choice(n_bins, p=masses)
        rand_x = [x[rand_bin], x[rand_bin + 1]]
        rand_y = [y[rand_bin], y[rand_bin + 1]]
        #
        # get a sample from it
        s = None
        if sampling == 'rejection':
            s = rejection_sampling_from_trapezoid(rand_x, rand_y, n_samples=1, rand_gen=rand_gen)
            samples.append(s[0])
        else:
            raise ValueError('Only rejection sampling implemented')

    return numpy.array(samples)
