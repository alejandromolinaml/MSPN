from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import logging
import argparse
try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time
import datetime
import os
import sys
import pickle
import gzip
import inspect

import numpy
# from numpy.testing import assert_almost_equal

from mlutils.datasets import loadMLC

from tfspn.SPN import Splitting
from tfspn.SPN import SPN, Splitting

# from experiments.mnist.augmenting_mnist import decode_predictions
# from experiments.mnist.augmenting_mnist import plot_digit
# from experiments.mnist.augmenting_mnist import load_ae_decoder

from sklearn.model_selection import StratifiedKFold

# import theano


# def get_nearest_neighbours_theano_func():
#     """
#     Returns the id of the nearest instance to sample and its value,
#     in the euclidean distance sense
#     """

#     sample = theano.tensor.vector(dtype=theano.config.floatX)
#     data = theano.tensor.matrix(dtype=theano.config.floatX)

#     distance_vec = theano.tensor.sum((data - sample) ** 2, axis=1)
#     nn_id = theano.tensor.argmin(distance_vec)

#     find_nearest_neighbour = theano.function(inputs=[sample, data],
#                                              outputs=[nn_id, data[nn_id]])
#     return find_nearest_neighbour


# def get_nearest_neighbour(samples, data, nn_func=None):

#     if nn_func is None:
#         nn_func = get_nearest_neighbours_theano_func()
#         data = data.astype(theano.config.floatX)
#         samples = [s.astype(theano.config.floatX) for s in samples]

#     neighbours = []

#     for instance in samples:
#         nn_s_t = perf_counter()

#         # else:
#         nn_id, instance_nn = nn_func(instance, data)
#         nn_e_t = perf_counter()
#         print(data.shape)
#         neighbours.append((nn_id, data[nn_id]))
#         logging.info('Got nn {} in {} secs'.format(nn_id, nn_e_t - nn_s_t))
#     return neighbours


ROW_SPLIT_METHODS = {
    'kmeans': Splitting.KmeansRows,
    'rdc-kmeans': Splitting.KmeansRDCRows,
    'rand-cond': Splitting.RandomPartitionConditioningRows,
    'rand-mode-split': Splitting.RandomBinaryModalSplitting,
    'rand-split': Splitting.RandomBalancedBinarySplit,
    'dbscan': Splitting.DBScan,
    'gower': Splitting.Gower,
}

COL_SPLIT_METHODS = {
    'rdc': Splitting.RDCTest,  # FIXME: put the new dcor method
    'gdt': Splitting.GDTTest,
    'ind': Splitting.IndependenceTest
}

DEFAULT_ROW_SPLIT_ARGS = {
    'n_clusters': {'default': 2, 'type': int},
    'OHE': {'default': 1, 'type': bool},
    'eps': {'default': 0.3, 'type': float},
    'min_samples': {'default': 10, 'type': int},
    'k': {'default': 20, 'type': int},
    's': {'default': 1. / 6., 'type': float},
    # 'non_linearity': numpy.sin
}

DEFAULT_COL_SPLIT_ARGS = {
    'threshold': {'default': 0.3, 'type': float},
    'alpha': {'default': 0.001, 'type': float},
    'OHE': {'default': 1, 'type': bool},
    'linear': {'default': 1, 'type': bool},

}


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='(MLC) dataset name')

parser.add_argument("--data-dir", type=str, nargs='?',
                    default='mlutils/datasets/MLC/proc-db/proc/',
                    help='Specify dataset dir (default data/)')

# parser.add_argument('--train-ext', type=str,
#                     default=None,
#                     help='Training set extension')

# parser.add_argument('--valid-ext', type=str,
#                     default=None,
#                     help='Validation set extension')

# parser.add_argument('--test-ext', type=str,
#                     default=None,
#                     help='Test set extension')

# parser.add_argument('-k', '--n-row-clusters', type=int, nargs='?',
#                     default=2,
#                     help='Number of clusters to split rows into')

parser.add_argument('-r', '--row-split', type=str, nargs='?',
                    default='kmeans',
                    help='Cluster method to apply on rows')

parser.add_argument('-c', '--col-split', type=str, nargs='?',
                    default='rdc-cancor',
                    help='(In)dependency test to apply to columns')

parser.add_argument('--row-split-args', type=str, nargs='?',
                    help='Additional row split method parameters in the form of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('--col-split-args', type=str, nargs='?',
                    help='Additional col split method parameters in the form of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('--exp-id', type=int,
                    help='Experiment id (it will be the row in the memory map table)')

parser.add_argument('--memmap', type=str, nargs='?',
                    default=None,
                    help='Memory map dir')

parser.add_argument('--leaf', type=str, nargs='?',
                    default='piecewise',
                    help='Leaf type distribution (piecewise|kde|isotonic|histogram)')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')

# parser.add_argument('-t', '--col-split-threshold', type=float, nargs='?',
#                     default=0.75,
#                     help='Threshold value for column dependency clustering')

parser.add_argument('-m', '--min-inst-slice', type=int, nargs='?',
                    default=50,
                    help='Min number of instances in a slice to split by cols')

parser.add_argument('--cv', type=int, nargs='?',
                    default=None,
                    help='If to apply cross validation and how many folds')

parser.add_argument('--target-id', type=int, nargs='?',
                    default=-1,
                    help='Id of target feature, if any, used for stratified k fold cross validation')

parser.add_argument('--tail-width', type=float, nargs='?',
                    default=1,
                    help='Tail width')

parser.add_argument('-a', '--alpha', type=float, nargs='?',
                    default=1.0,
                    help='Smoothing factor for leaf probability estimation')

parser.add_argument('--ohe-cat-rows', action='store_true',
                    help='Whether to use one hot encoding for categorical rows')

parser.add_argument('--ohe-noise', type=float, nargs='?',
                    default=None,
                    help='Gaussian noise var to apply to one hot encoded categorical vars')

parser.add_argument('--isotonic', action='store_true',
                    help='Whether to use isotonic regression for piecewise leaves')

parser.add_argument('--bootstraps', type=int, nargs='?',
                    default=None,
                    help='Number of bootstrap histograms to smooth piecewise leaves')

parser.add_argument('--average-bootstraps', action='store_true',
                    help='Whether to average bootstrapped histograms for piecewise leaves')

parser.add_argument('--product-first', action='store_true',
                    help='Whether to split first on the columns')

parser.add_argument('--save-model', action='store_true',
                    help='Whether to store the model file as a pickle file')

parser.add_argument('--gzip', action='store_true',
                    help='Whether to compress the model pickle file')

parser.add_argument('--kernel-family', type=str, nargs='?',
                    default='gaussian',
                    help='Leaf KDE nodes kernel family')

parser.add_argument('--kernel-bandwidth', type=float, nargs='?',
                    default=0.2,
                    help='Leaf KDE nodes kernel bandwidth')

parser.add_argument('--kernel-metric', type=str, nargs='?',
                    default='euclidean',
                    help='Leaf KDE nodes kernel metric')

parser.add_argument('--prior-weight', type=float, nargs='?',
                    default=0.01,
                    help='Prior weight for mixing box uniform distribution')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')
#
# parsing the args
args = parser.parse_args()

#
# setting verbosity level
logger = logging.getLogger()
if args.verbose == 1:
    # logger.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.setLevel(logging.INFO)
elif args.verbose == 2:
    # logger.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

rand_gen = numpy.random.RandomState(args.seed)

row_split_args = None
if args.row_split_args is not None:
    row_key_value_pairs = args.row_split_args.translate(
        {ord('['): '', ord(']'): ''}).split(',')
    row_split_args = {key.strip(): value.strip() for key, value in
                      [pair.strip().split('=')
                       for pair in row_key_value_pairs]}
else:
    row_split_args = {}
logging.info('Row split method parameters:  {}'.format(row_split_args))

col_split_args = None
if args.col_split_args is not None:
    col_key_value_pairs = args.col_split_args.translate(
        {ord('['): '', ord(']'): ''}).split(',')
    col_split_args = {key.strip(): value.strip() for key, value in
                      [pair.strip().split('=')
                       for pair in col_key_value_pairs]}
else:
    col_split_args = {}
logging.info('Col split method parameters:  {}'.format(col_split_args))
logger.info("Starting with arguments:\n{}\n\n".format(args))

#
# loading the MLC datasets
# TODO: extend to other datasets
dataset_name = args.dataset
logging.info('Looking for dataset {} ...in dir {}'.format(dataset_name, args.data_dir))
(train, valid, test), feature_names, feature_types, domains = loadMLC(dataset_name,
                                                                      base_path='',
                                                                      data_dir=args.data_dir)
logging.info('Loaded\n\ttrain:\t{}\n\tvalid:\t{}\n\ttest:\t{}'.format(train.shape,
                                                                      valid.shape,
                                                                      test.shape))

#
# creating output paths and dirs, if not present
out_path = os.path.join(args.output, dataset_name, str(args.exp_id))
logging.info('Opening logging file in {}'.format(out_path))
os.makedirs(out_path, exist_ok=True)
out_log_path = os.path.join(out_path, 'exp.log')

#
# memory mapping
mem_map = None
if args.memmap:
    mem_map = numpy.memmap(args.memmap, dtype='float', mode='r+')

#
# learning the spn
cluster_first = not args.product_first
print('CLUSTER FIRST', cluster_first)

DEFAULT_ROW_SPLIT_ARGS['seed'] = {'default': args.seed, 'type': int}

#
# getting the splitting functions
row_split_func = ROW_SPLIT_METHODS[args.row_split]
row_split_arg_names = inspect.getargspec(row_split_func)
print('row args', row_split_arg_names[0])

r_args = {}
for r_arg in row_split_arg_names[0]:
    r_val = None
    if r_arg not in row_split_args:
        r_val = DEFAULT_ROW_SPLIT_ARGS[r_arg]['default']
    else:
        r_val = row_split_args[r_arg]
    r_args[r_arg] = DEFAULT_ROW_SPLIT_ARGS[r_arg]['type'](r_val)

print('\n\n\nR ARGS', r_args)
row_split_method = row_split_func(**r_args)


col_split_func = COL_SPLIT_METHODS[args.col_split]
col_split_arg_names = inspect.getargspec(col_split_func)
print('col args', col_split_arg_names[0])

c_args = {}
for c_arg in col_split_arg_names[0]:
    c_val = None
    if c_arg not in col_split_args:
        c_val = DEFAULT_COL_SPLIT_ARGS[c_arg]['default']
    else:
        c_val = col_split_args[c_arg]
    c_args[c_arg] = DEFAULT_COL_SPLIT_ARGS[c_arg]['type'](c_val)

print('\n\n\nC ARGS', c_args)
col_split_method = col_split_func(**c_args)


# families = None
families = [args.leaf] * train.shape[1]

fold_splits = None
n_folds = 1 if args.cv is None else args.cv
if args.cv is not None:

    target_rv = args.target_id
    logging.info('Performing {}-fold cross validation (stratified on attribute {})'.format(args.cv,
                                                                                           target_rv))
    whole_data = numpy.concatenate([train, valid, test], axis=0)
    logging.info('\tWhole data: {}'.format(whole_data.shape))

    skf = StratifiedKFold(n_splits=args.cv)
    fold_splits = [(whole_data[train_index], None, whole_data[test_index])
                   for train_index, test_index in skf.split(whole_data, whole_data[:, target_rv])]

else:
    fold_splits = [(train, valid, test)]

fold_train_lls = []
fold_valid_lls = []
fold_test_lls = []
#
#
# preparing output configuration
table_header = '\t'.join(['dataset',
                          'fold',
                          'row-split',
                          'row-split-args',
                          'col-split',
                          'col-split-args',
                          'min-inst-split',
                          'alpha',
                          'leaf',
                          'prior_weight',
                          # 'bootstraps',
                          # 'avg-bootstraps',
                          # 'prod-first',
                          'train-avg-ll',
                          'valid-avg-ll',
                          'test-avg-ll',
                          'learning-time',
                          'eval-time',
                          'nodes',
                          'edges',
                          'layers',
                          'prod-nodes',
                          'sum-nodes',
                          'leaves',
                          'spn-json-path',
                          'lls-files'
                          ])
table_header += '\n'
with open(out_log_path, 'w') as out_file:
    out_file.write(table_header)

    for f, (train, valid, test) in enumerate(fold_splits):

        logging.info('\n\n######## FOLD {}/{} ###########\n'.format(f + 1, n_folds))
        valid_str = None if valid is None else valid.shape
        logging.info('train:{} valid:{} test:{}'.format(train.shape,
                                                        valid_str,
                                                        test.shape))
        spn_json_path = os.path.join(out_path, 'spn.{}.json'.format(f))

        # train = whole_data[train_index]
        # test = whole_data[test_index]

        learn_start_t = perf_counter()
        spn = SPN.LearnStructure(train,
                                 featureNames=feature_names,
                                 families=families,
                                 domains=domains,
                                 featureTypes=feature_types,
                                 min_instances_slice=args.min_inst_slice,
                                 bin_width=args.tail_width,
                                 alpha=args.alpha,
                                 isotonic=args.isotonic,
                                 pw_bootstrap=args.bootstraps,
                                 avg_pw_boostrap=args.average_bootstraps,
                                 row_split_method=row_split_method,
                                 col_split_method=col_split_method,
                                 cluster_first=cluster_first,
                                 kernel_family=args.kernel_family,
                                 kernel_bandwidth=args.kernel_bandwidth,
                                 kernel_metric=args.kernel_metric,
                                 prior_weight=args.prior_weight,
                                 rand_seed=args.seed
                                 )
        learn_end_t = perf_counter()
        learning_time = learn_end_t - learn_start_t
        logging.info('\n\n*****Spn learned in {} secs! *****'.format(learning_time))

        logging.info('Learned spn:\n{}'.format(spn))
        # spn.ToFile(spn_json_path)
        # logging.info('spn dumped to json file {}'.format(spn_json_path))

        spn_pickle_path = os.path.join(out_path, 'spn.{}.pklz'.format(f))
        SPN.to_pickle(spn, spn_pickle_path)
        logging.info('spn dumped to pickle file {}'.format(spn_pickle_path))

        logging.info('\n\nstructure stats:')
        n_nodes = spn.n_nodes()
        logging.info('# nodes {}'.format(n_nodes))
        n_sum_nodes = spn.n_sum_nodes()
        logging.info('\t# sum nodes {}'.format(n_sum_nodes))
        n_prod_nodes = spn.n_prod_nodes()
        logging.info('\t# prod nodes {}'.format(n_prod_nodes))
        n_leaves = spn.n_leaves()
        logging.info('\t# leaf nodes {}'.format(n_leaves))
        n_edges = spn.n_edges()
        logging.info('# edges {}'.format(n_edges))
        n_layers = spn.n_layers()
        logging.info('# layers {}'.format(n_layers))

        #
        #
        # evaluating on likelihood
        logging.info('\n\nlikelihood:')

        eval_start_t = perf_counter()

        train_lls = spn.root.eval(train)
        valid_lls = None
        if valid is not None:
            valid_lls = spn.root.eval(valid)
        test_lls = spn.root.eval(test)

        eval_end_t = perf_counter()
        eval_time = eval_end_t - eval_start_t
        logging.info('\n\nFOLD  {}/{}'.format(f + 1, n_folds))
        logging.info('\n\n*****Spn eval in {} secs! *****'.format(eval_time))

        train_avg_ll = numpy.mean(train_lls)
        logging.info("\ttrain:\t{}\t(min:{}\tmax:{})".format(
            train_avg_ll, train_lls.min(), train_lls.max()))
        fold_train_lls.append(train_avg_ll)

        valid_avg_ll = None
        if valid is not None:
            valid_avg_ll = numpy.mean(valid_lls)
            logging.info("\tvalid:\t{}\t(min:{}\tmax:{})".format(
                valid_avg_ll, valid_lls.min(), valid_lls.max()))
            fold_valid_lls.append(valid_avg_ll)

        test_avg_ll = numpy.mean(test_lls)
        logging.info("\ttest:\t{}\t(min:{}\tmax:{})".format(
            test_avg_ll, test_lls.min(), test_lls.max()))
        fold_test_lls.append(test_avg_ll)

        exp_str = dataset_name
        exp_str += '\t{}'.format(f)
        exp_str += '\t{}'.format(args.row_split)
        exp_str += '\t{}'.format(args.row_split_args)
        exp_str += '\t{}'.format(args.col_split)
        exp_str += '\t{}'.format(args.col_split_args)
        exp_str += '\t{}'.format(args.min_inst_slice)
        exp_str += '\t{}'.format(args.alpha)
        exp_str += '\t{}'.format(args.leaf)
        exp_str += '\t{}'.format(args.prior_weight)
        # exp_str += '\t{}'.format(args.bootstraps)
        # exp_str += '\t{}'.format(args.average_bootstraps)
        # exp_str += '\t{}'.format(args.product_first)
        exp_str += '\t{}'.format(train_avg_ll)
        exp_str += '\t{}'.format(valid_avg_ll)
        exp_str += '\t{}'.format(test_avg_ll)
        exp_str += '\t{}'.format(learning_time)
        exp_str += '\t{}'.format(eval_time)
        exp_str += '\t{}'.format(n_nodes)
        exp_str += '\t{}'.format(n_edges)
        exp_str += '\t{}'.format(n_layers)
        exp_str += '\t{}'.format(n_prod_nodes)
        exp_str += '\t{}'.format(n_sum_nodes)
        exp_str += '\t{}'.format(n_leaves)
        exp_str += '\t{}'.format(spn_json_path)
        lls_file_path = os.path.join(out_path, 'lls-{}'.format(f))
        exp_str += '\t{}'.format(lls_file_path)
        exp_str += '\n'
        out_file.write(exp_str)

        # saving lls to numpy files
        numpy.save('{}.train'.format(lls_file_path), train_lls)
        if valid is not None:
            numpy.save('{}.valid'.format(lls_file_path), valid_lls)
        numpy.save('{}.test'.format(lls_file_path), test_lls)

        #
        # saving to mem_map
        if mem_map is not None:
            n_cols = 13
            mem_map = mem_map.reshape(-1, n_cols)
            print('mm', mem_map.shape)
            config_array = numpy.array([args.exp_id,
                                        f,
                                        train_avg_ll, valid_avg_ll, test_avg_ll,
                                        learning_time, eval_time, n_nodes, n_edges,
                                        n_layers, n_prod_nodes, n_sum_nodes,
                                        n_leaves])
            print('config', config_array.shape)
            mem_map[args.exp_id + f, :] = config_array
            print('mm', mem_map.shape)
            mem_map.flush()
            logging.info('Memmap {} flushed'.format(args.memmap))

    logging.info('FOLDS ENDED')
    fold_train_ll_avg = numpy.mean(fold_train_lls)
    fold_train_ll_std = numpy.std(fold_train_lls)
    logging.info('Average train ll {}:\t+/- {}'.format(fold_train_ll_avg,
                                                       fold_train_ll_std))
    fold_valid_ll_avg, fold_valid_ll_std = None, None
    if len(fold_valid_lls) > 0:
        fold_valid_ll_avg = numpy.mean(fold_valid_lls)
        fold_valid_ll_std = numpy.std(fold_valid_lls)
        logging.info('Average valid ll {}:\t+/- {}'.format(fold_valid_ll_avg,
                                                           fold_valid_ll_std))

    fold_test_ll_avg = numpy.mean(fold_test_lls)
    fold_test_ll_std = numpy.std(fold_test_lls)
    logging.info('Average test ll {}:\t+/- {}'.format(fold_test_ll_avg,
                                                      fold_test_ll_std))

    out_file.write('{}\t{}\t{}\t{}\t{}\t{}'.format(fold_train_ll_avg,
                                                   fold_train_ll_std,
                                                   fold_valid_ll_avg,
                                                   fold_valid_ll_std,
                                                   fold_test_ll_avg,
                                                   fold_test_ll_std))


# learn_start_t = perf_counter()
# spn = SPN.LearnStructure(train,
#                          featureNames=feature_names,
#                          families=families,
#                          domains=domains,
#                          featureTypes=feature_types,
#                          min_instances_slice=args.min_inst_slice,
#                          bin_width=args.tail_width,
#                          alpha=args.alpha,
#                          isotonic=args.isotonic,
#                          pw_bootstrap=args.bootstraps,
#                          avg_pw_boostrap=args.average_bootstraps,
#                          row_split_method=row_split_method,
#                          col_split_method=col_split_method,
#                          cluster_first=cluster_first,
#                          kernel_family=args.kernel_family,
#                          kernel_bandwidth=args.kernel_bandwidth,
#                          kernel_metric=args.kernel_metric,
#                          rand_seed=args.seed
#                          )
# learn_end_t = perf_counter()
# learning_time = learn_end_t - learn_start_t
# logging.info('\n\n*****Spn learned in {} secs! *****'.format(learning_time))

# logging.info('Learned spn:\n{}'.format(spn))
# spn.ToFile(spn_json_path)
# logging.info('spn dumped to json file {}'.format(spn_json_path))

# spn_pickle_path = os.path.join(out_path, 'spn.pklz')
# SPN.to_pickle(spn, spn_pickle_path)
# logging.info('spn dumped to pickle file {}'.format(spn_pickle_path))

# logging.info('\n\nstructure stats:')
# n_nodes = spn.n_nodes()
# logging.info('# nodes {}'.format(n_nodes))
# n_sum_nodes = spn.n_sum_nodes()
# logging.info('\t# sum nodes {}'.format(n_sum_nodes))
# n_prod_nodes = spn.n_prod_nodes()
# logging.info('\t# prod nodes {}'.format(n_prod_nodes))
# n_leaves = spn.n_leaves()
# logging.info('\t# leaf nodes {}'.format(n_leaves))
# n_edges = spn.n_edges()
# logging.info('# edges {}'.format(n_edges))
# n_layers = spn.n_layers()
# logging.info('# layers {}'.format(n_layers))


# #
# #
# # evaluating on likelihood
# logging.info('\n\nlikelihood:')

# eval_start_t = perf_counter()

# train_lls = spn.root.eval(train)
# valid_lls = spn.root.eval(valid)
# test_lls = spn.root.eval(test)


# eval_end_t = perf_counter()
# eval_time = eval_end_t - eval_start_t
# logging.info('\n\n*****Spn eval in {} secs! *****'.format(eval_time))

# train_avg_ll = numpy.mean(train_lls)
# valid_avg_ll = numpy.mean(valid_lls)
# test_avg_ll = numpy.mean(test_lls)
# logging.info("\ttrain:\t{}\t(min:{}\tmax:{})".format(
#     train_avg_ll, train_lls.min(), train_lls.max()))
# logging.info("\tvalid:\t{}\t(min:{}\tmax:{})".format(
#     valid_avg_ll, valid_lls.min(), valid_lls.max()))
# logging.info("\ttest:\t{}\t(min:{}\tmax:{})".format(test_avg_ll, test_lls.min(), test_lls.max()))

# #
# #
# # preparing output configuration

# table_header = '\t'.join(['dataset',
#                           'row-split',
#                           'row-split-args',
#                           'col-split',
#                           'col-split-args',
#                           'min-inst-split',
#                           'alpha',
#                           'isotonic',
#                           'bootstraps',
#                           'avg-bootstraps',
#                           'prod-first',
#                           'train-avg-ll',
#                           'valid-avg-ll',
#                           'test-avg-ll',
#                           'learning-time',
#                           'eval-time',
#                           'nodes', 'edges', 'layers',
#                           'prod-nodes', 'sum-nodes', 'leaves',
#                           'spn-json-path',
#                           'lls-files'
#                           ])
# table_header += '\n'

# exp_str = dataset_name
# exp_str += '\t{}'.format(args.row_split)
# exp_str += '\t{}'.format(args.row_split_args)
# exp_str += '\t{}'.format(args.col_split)
# exp_str += '\t{}'.format(args.col_split_args)
# exp_str += '\t{}'.format(args.min_inst_slice)
# exp_str += '\t{}'.format(args.alpha)
# exp_str += '\t{}'.format(args.isotonic)
# exp_str += '\t{}'.format(args.bootstraps)
# exp_str += '\t{}'.format(args.average_bootstraps)
# exp_str += '\t{}'.format(args.product_first)
# exp_str += '\t{}'.format(train_avg_ll)
# exp_str += '\t{}'.format(valid_avg_ll)
# exp_str += '\t{}'.format(test_avg_ll)
# exp_str += '\t{}'.format(learning_time)
# exp_str += '\t{}'.format(eval_time)
# exp_str += '\t{}'.format(n_nodes)
# exp_str += '\t{}'.format(n_edges)
# exp_str += '\t{}'.format(n_layers)
# exp_str += '\t{}'.format(n_prod_nodes)
# exp_str += '\t{}'.format(n_sum_nodes)
# exp_str += '\t{}'.format(n_leaves)
# exp_str += '\t{}'.format(spn_json_path)
# lls_file_path = os.path.join(out_path, 'lls')
# exp_str += '\t{}'.format(lls_file_path)

# #
# saving lls to numpy files
# numpy.save('{}.train'.format(lls_file_path), train_lls)
# numpy.save('{}.valid'.format(lls_file_path), valid_lls)
# numpy.save('{}.test'.format(lls_file_path), test_lls)

# #
# # mpe
# mpe_data = numpy.copy(train[:10])
# print("train data\n", mpe_data)
# mpe_data[:, :10] = numpy.nan
# print("Masked data\n", mpe_data)

# one_more_row = numpy.array([numpy.nan] * 10 + [1, 1, 0, 0, 0])
# one_more_row = one_more_row.reshape(-1, one_more_row.shape[0])
# print(one_more_row)
# mpe_data = numpy.concatenate([mpe_data,
#                               one_more_row], axis=0)

# mpe_probs, mpe_res = spn.root.mpe_eval(mpe_data)
# print("MPE\n", mpe_res)
# numpy.save('mpe-ass', mpe_res)


# mpe_embeddings = mpe_res[:, :10]
# print(mpe_embeddings.shape)

# # for t in train:
# #     for m in mpe_embeddings:
# #         if numpy.all(numpy.isclose(m[:10], t[:10])):
# #             print('close', m, t)
# #         for i in range(10):
# #             if numpy.isclose(m[i], t[i]):
# #                 print('close', m[i], t[i])

# dec_path = '/home/valerio/Petto Redigi/trash/raelk.mnist_2017-05-22_15-00-18/ae.raelk.mnist.decoder.0'
# decoder = load_ae_decoder(dec_path)
# print('loaded the decoder from {}'.format(dec_path))
# print(decoder)
# dec_imgs = decode_predictions(mpe_embeddings, decoder)
# # print('decoded samples {}\n{}'.format(dec_imgs.shape, dec_imgs))

# nn_imgs = get_nearest_neighbour(dec_imgs, train[:, :10])

# for i, img in enumerate(dec_imgs):
#     print(mpe_res[i])
#     plot_digit(img)

#     plot_digit(nn_imgs[i])

# #
# # saving to mem_map
# if mem_map is not None:
#     n_cols = 12
#     mem_map = mem_map.reshape(-1, n_cols)
#     print('mm', mem_map.shape)
#     mem_map[args.exp_id, :] = numpy.array([args.exp_id,
#                                            train_avg_ll, valid_avg_ll, test_avg_ll,
#                                            learning_time, eval_time, n_nodes, n_edges,
#                                            n_layers, n_prod_nodes, n_sum_nodes,
#                                            n_leaves])
#     mem_map.flush()

# exp_str += '\n'
# with open(out_log_path, 'w') as out_file:
#     out_file.write(table_header)
#     out_file.write(exp_str)
