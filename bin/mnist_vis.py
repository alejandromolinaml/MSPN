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
import itertools

import numpy
# from numpy.testing import assert_almost_equal

from mlutils.datasets import loadMLC

from tfspn.SPN import Splitting
from tfspn.SPN import SPN, Splitting

from experiments.mnist.augmenting_mnist import decode_predictions
from experiments.mnist.augmenting_mnist import plot_digit
from experiments.mnist.augmenting_mnist import plot_digits_matrix
from experiments.mnist.augmenting_mnist import plot_embedding
from experiments.mnist.augmenting_mnist import load_ae_decoder
from experiments.mnist.augmenting_mnist import load_train_val_test_splits
from experiments.mnist.augmenting_mnist import encode_predictions
from experiments.mnist.augmenting_mnist import load_ae_encoder
from experiments.mnist.augmenting_mnist import get_image_mask
from experiments.mnist.augmenting_mnist import RED_CMAP, BLUE_CMAP

from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import average_precision_score
from sklearn import manifold


def compute_scores(y_true, y_preds, score='accuracy'):

    if score == 'accuracy':
        return accuracy_score(y_true, y_preds)
    elif score == 'hamming':
        return 1 - hamming_loss(y_true, y_preds)
    elif score == 'exact':
        return 1 - zero_one_loss(y_true, y_preds)
    elif score == 'jaccard':
        return jaccard_similarity_score(y_true, y_preds)
    elif score == 'micro-f1':
        return f1_score(y_true, y_preds, average='micro')
    elif score == 'macro-f1':
        return f1_score(y_true, y_preds, average='macro')
    elif score == 'micro-auc-pr':
        return average_precision_score(y_true, y_preds, average='micro')
    elif score == 'macro-auc-pr':
        return average_precision_score(y_true, y_preds, average='macro')


def get_nearest_neighbours_theano_func():
    """
    Returns the id of the nearest instance to sample and its value,
    in the euclidean distance sense
    """

    import theano
    sample = theano.tensor.vector(dtype=theano.config.floatX)
    data = theano.tensor.matrix(dtype=theano.config.floatX)

    distance_vec = theano.tensor.sum((data - sample) ** 2, axis=1)
    nn_id = theano.tensor.argmin(distance_vec)

    find_nearest_neighbour = theano.function(inputs=[sample, data],
                                             outputs=[nn_id, data[nn_id]])
    return find_nearest_neighbour


def get_nearest_neighbours_numpy_func(sample, data):

    distance_vec = numpy.sum((data - sample) ** 2, axis=1)
    nn_id = numpy.argmin(distance_vec)

    return nn_id, data[nn_id]


def get_nearest_neighbour(samples, data, nn_func=None):

    if nn_func is None:
        # nn_func = get_nearest_neighbours_theano_func()
        # data = data.astype(theano.config.floatX)
        # samples = [s.astype(theano.config.floatX) for s in samples]
        nn_func = get_nearest_neighbours_numpy_func

    neighbours = []

    for instance in samples:
        nn_s_t = perf_counter()

        # else:
        nn_id, instance_nn = nn_func(instance, data)
        nn_e_t = perf_counter()
        print(data.shape)
        neighbours.append((nn_id, data[nn_id]))
        logging.info('Got nn {} in {} secs'.format(nn_id, nn_e_t - nn_s_t))
    return neighbours


def binary_labels_to_int(bin_labels):
    bin_labels = numpy.array(bin_labels).astype(numpy.int32)
    powers = numpy.power(2, bin_labels)
    return powers.sum(axis=1)


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='(MLC) dataset name')

parser.add_argument("--data-dir", type=str, nargs='?',
                    default='./mlutils/datasets/mnist/',
                    help='Specify dataset dir (default data/)')

parser.add_argument('--orig-data', type=str,
                    default='raelk.mnist.pklz',
                    help='original data path (e.g. full mnist in pickle file)')

# parser.add_argument('--valid-ext', type=str,
#                     default=None,
#                     help='Validation set extension')

# parser.add_argument('--test-ext', type=str,
#                     default=None,
#                     help='Test set extension')

# parser.add_argument('-k', '--n-row-clusters', type=int, nargs='?',
#                     default=2,
#                     help='Number of clusters to split rows into')

parser.add_argument('-s', '--spn', type=str,
                    help='Spn pickle path')

parser.add_argument('--ae-path', type=str, nargs='?',
                    default='/home/valerio/Petto Redigi/trash/raelk.mnist_2017-05-22_15-00-18/',
                    help='decoder path')


# parser.add_argument('-d', '--decoder', type=str, nargs='?',
#                     default='/home/valerio/Petto Redigi/trash/raelk.mnist_2017-05-22_15-00-18/ae.raelk.mnist.decoder.0',
#                     help='decoder path')

# parser.add_argument('-e', '--encoder', type=str, nargs='?',
#                     default='/home/valerio/Petto Redigi/trash/raelk.mnist_2017-05-22_15-00-18/ae.raelk.mnist.encoder.0',
#                     help='encoder path')


parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('--aug-features', type=int, nargs='?',
                    default=5,
                    help='Number of augmented features')

parser.add_argument('--emb-features', type=int, nargs='?',
                    default=10,
                    help='Number of embedded features')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')

parser.add_argument('--all-worlds', action='store_true',
                    help='Compute MPE and visualize for all possible label combinations')

parser.add_argument('--mpe-labels', action='store_true',
                    help='Predicting labels for test data')

parser.add_argument('--combine', action='store_true',
                    help='Combine image')

parser.add_argument('--vis-2d', action='store_true',
                    help='visualize embedding as 2d images')

parser.add_argument('--show-images', action='store_true',
                    help='Show single images')

parser.add_argument('--mpe-embeddings', action='store_true',
                    help='Show MPE predictions for embeddings')

parser.add_argument('--mpe-class', action='store_true',
                    help='Class variable is present')

parser.add_argument('--marginalize-labels', action='store_true',
                    help='Marginalizing labels, class variable is present')

parser.add_argument('--privileged-inference', action='store_true',
                    help='All inference results for privileged case')

parser.add_argument('--filter-samples-prob', type=int, nargs='?',
                    default=None,
                    help='All inference results for privileged case')

parser.add_argument('--marginal-ll-emb', action='store_true',
                    help='Compute the marginal log likelihood on the embedding features only')

# parser.add_argument('-t', '--col-split-threshold', type=float, nargs='?',
#                     default=0.75,
#                     help='Threshold value for column dependency clustering')

parser.add_argument('--n-images', type=int,
                    default=64,
                    help='Number of images to visualize')

parser.add_argument('--n-rows', type=int,
                    default=None,
                    help='Number of image rows to visualize')

parser.add_argument('--n-cols', type=int,
                    default=None,
                    help='Number of image columns to visualize')

parser.add_argument('--exp-id', type=int,
                    help='Experiment id (it will be the row in the memory map table)')

parser.add_argument('--memmap', type=str, nargs='?',
                    default=None,
                    help='Memory map dir')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')

parser.add_argument('--samples', type=int, nargs='?',
                    default=None,
                    help='Seed for the random generator')

parser.add_argument("--masks", type=str, nargs='+',
                    # default=['up', 'down', 'left', 'right'],
                    default=[],
                    help='Specify which masked data to create')


parser.add_argument("--mask-emb-features", type=int, nargs='+',
                    default=None,
                    help='Specify how many features for each mask')

parser.add_argument('--img-size', type=int, nargs='+',
                    default=[],
                    help='For image data, specify the size of the image for reshaping purposes')


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

logger.info("Starting with arguments:\n%s", args)

os.makedirs(args.output, exist_ok=True)

#
#
# load original mnist
orig_data_path = os.path.join(args.data_dir, args.orig_data)
orig_fold_splits = load_train_val_test_splits(orig_data_path,
                                              'mnist',
                                              train_ext=None,
                                              valid_ext=None,
                                              test_ext=None,
                                              x_only=False,
                                              y_only=False,
                                              dtype=numpy.float64)

orig_train, orig_valid, orig_test = orig_fold_splits[0]

orig_train_x, orig_train_y = orig_train
orig_valid_x, orig_valid_y = orig_valid
orig_test_x, orig_test_y = orig_test


#
# load augmented mnist in MLC format
dataset_name = args.dataset
logging.info('Looking for dataset {} ...in dir {}'.format(dataset_name, args.data_dir))
(train, valid, test), feature_names, feature_types, domains = loadMLC(dataset_name,
                                                                      base_path='',
                                                                      data_dir=args.data_dir)
logging.info('Loaded\n\ttrain:\t{}\n\tvalid:\t{}\n\ttest:\t{}'.format(train.shape,
                                                                      valid.shape,
                                                                      test.shape))


load_start_t = perf_counter()
# spn = SPN.FromFile(args.spn)
spn = SPN.from_pickle(args.spn)
# spn = None
load_end_t = perf_counter()
# print(spn)
logging.info('spn loaded from pickle in {} secs'.format(load_end_t - load_start_t))


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
# n_layers = spn.n_layers()
# logging.info('# layers {}'.format(n_layers))


# #
# # evaluating on likelihood
# logging.info('\n\nlikelihood:')
# train_lls = spn.root.eval(train)
# valid_lls = spn.root.eval(valid)
# test_lls = spn.root.eval(test)

# train_avg_ll = numpy.mean(train_lls)
# valid_avg_ll = numpy.mean(valid_lls)
# test_avg_ll = numpy.mean(test_lls)
# logging.info("\ttrain:\t{}".format(train_avg_ll))
# logging.info("\tvalid:\t{}".format(valid_avg_ll))
# logging.info("\ttest:\t{}".format(test_avg_ll))


#
# mpe

n_aug_features = args.aug_features
print('# AUG features', n_aug_features)
n_emb_features = args.emb_features
print('# EMB features', n_emb_features)

MASKED_IMAGE_SIZE = {'left': (28, 14),
                     'right': (28, 14),
                     'up': (14, 28),
                     'down': (14, 28)}

if args.privileged_inference:

    #
    # memory mapping
    mem_map = None
    if args.memmap:
        mem_map = numpy.memmap(args.memmap, dtype='float', mode='r+')

    out_log_path = os.path.join(args.output, 'exp.log')
    header = '\t'.join(['exp-id',
                        'train-joint-ll', 'valid-joint-ll', 'test-joint-ll',
                        'train-marg-ll', 'valid-marg-ll', 'test-marg-ll',
                        'class-mpe-acc',
                        'class-map-acc',
                        'labels-mpe-jac',
                        'labels-mpe-jam',
                        'labels-mpe-exa'])

    with open(out_log_path, 'w') as out_log:
        out_log.write(header)
        out_log.write('\n')
        out_log.flush()

        #
        # joint likelihood
        logging.info('\n\n*** Joint log likelihood: *****')
        train_lls = spn.root.eval(train)
        valid_lls = spn.root.eval(valid)
        test_lls = spn.root.eval(test)

        train_avg_ll = numpy.mean(train_lls)
        logging.info("\ttrain:\t{}\t(min:{}\tmax:{})".format(
            train_avg_ll, train_lls.min(), train_lls.max()))

        valid_avg_ll = numpy.mean(valid_lls)
        logging.info("\tvalid:\t{}\t(min:{}\tmax:{})".format(
            valid_avg_ll, valid_lls.min(), valid_lls.max()))

        test_avg_ll = numpy.mean(test_lls)
        logging.info("\ttest:\t{}\t(min:{}\tmax:{})".format(
            test_avg_ll, test_lls.min(), test_lls.max()))

        #
        # marginalized likelihood over embeddings
        logging.info('\n\n*** Computing marginal log likelihood for embeddings ******')
        marg_out_rvs = {i for i in range(n_emb_features,   test.shape[1])}
        logging.info('RVs to marginalize {}'.format(marg_out_rvs))
        marg_spn = spn.root.marginalizeOut(marginals=marg_out_rvs)

        eval_start_t = perf_counter()
        marg_train_lls = marg_spn.eval(train)
        marg_valid_lls = marg_spn.eval(valid)
        marg_test_lls = marg_spn.eval(test)
        eval_end_t = perf_counter()
        eval_time = eval_end_t - eval_start_t
        logging.info('\n\n*****Spn eval in {} secs! *****'.format(eval_time))

        marg_train_avg_ll = numpy.mean(marg_train_lls)
        marg_valid_avg_ll = numpy.mean(marg_valid_lls)
        marg_test_avg_ll = numpy.mean(marg_test_lls)
        logging.info('\n\n')
        logging.info("\tmarg train:\t{}\t(min:{}\tmax:{})".format(marg_train_avg_ll,
                                                                  marg_train_lls.min(), marg_train_lls.max()))
        logging.info("\tmarg valid:\t{}\t(min:{}\tmax:{})".format(marg_valid_avg_ll,
                                                                  marg_valid_lls.min(), marg_valid_lls.max()))
        logging.info("\tmarg test:\t{}\t(min:{}\tmax:{})".format(marg_test_avg_ll,
                                                                 marg_test_lls.min(), marg_test_lls.max()))

        #
        # MPE class and labels predictions
        logging.info('\n\n*** Computing MPE predictions! ******')
        label_mpe_data = numpy.copy(test)
        label_mpe_data[:, n_emb_features:] = numpy.nan
        logging.info("Masked label data (first example)\n{}".format(label_mpe_data[0]))

        label_mpe_probs, label_mpe_res = spn.root.mpe_eval(label_mpe_data)
        logging.info("label MPE (first example)\n{}".format(label_mpe_res[0]))
        label_predictions_path = os.path.join(args.output, 'label-mpe-preds')
        numpy.save(label_predictions_path, label_mpe_res)

        #
        # MPE class attribute
        test_labels = test[:, n_emb_features].astype(numpy.int32)
        mpe_labels = label_mpe_res[:, n_emb_features].astype(numpy.int32)
        print(test_labels[:20], mpe_labels[:20])
        mpe_class_accuracy = compute_scores(test_labels, mpe_labels, 'accuracy')
        logging.info('MPE class accuracy {}'.format(mpe_class_accuracy))

        #
        # MPE labels
        test_labels = test[:, n_emb_features + 1:].astype(numpy.int32)
        mpe_labels = label_mpe_res[:, n_emb_features + 1:].astype(numpy.int32)

        mpe_labels_exa = compute_scores(test_labels, mpe_labels, 'exact')
        mpe_labels_ham = compute_scores(test_labels, mpe_labels, 'hamming')
        mpe_labels_jac = compute_scores(test_labels, mpe_labels, 'jaccard')
        logging.info('exact {}'.format(mpe_labels_exa))
        logging.info('hamming {}'.format(mpe_labels_ham))
        logging.info('jaccard {}'.format(mpe_labels_jac))

        #
        # MAP class
        logging.info('\n\n\n**** Marginalizing labels out (~class MAP) ******')

        marg_out_rvs = {i for i in range(n_emb_features + 1,   test.shape[1])}
        logging.info('RVs to marginalize {}'.format(marg_out_rvs))
        label_mpe_data = None
        label_map_data = numpy.copy(test)
        label_map_data[:, n_emb_features:] = numpy.nan
        map_marg_spn = spn.root.marginalizeOut(marginals=marg_out_rvs)
        label_map_probs, label_map_res = map_marg_spn.mpe_eval(label_map_data)

        mpe_labels = None
        test_labels = test[:, n_emb_features].astype(numpy.int32)
        map_labels = label_map_res[:, n_emb_features].astype(numpy.int32)
        print(test_labels[:20], map_labels[:20])
        map_class_accuracy = compute_scores(test_labels, map_labels, 'accuracy')
        logging.info('MAP class accuracy {}'.format(map_class_accuracy))

        #
        # writing to file
        exp_str = ''
        exp_str += '\t{}'.format(args.exp_id)
        exp_str += '\t{}'.format(train_avg_ll)
        exp_str += '\t{}'.format(valid_avg_ll)
        exp_str += '\t{}'.format(test_avg_ll)
        exp_str += '\t{}'.format(marg_train_avg_ll)
        exp_str += '\t{}'.format(marg_valid_avg_ll)
        exp_str += '\t{}'.format(marg_test_avg_ll)
        exp_str += '\t{}'.format(mpe_class_accuracy)
        exp_str += '\t{}'.format(map_class_accuracy)
        exp_str += '\t{}'.format(mpe_labels_jac)
        exp_str += '\t{}'.format(mpe_labels_ham)
        exp_str += '\t{}'.format(mpe_labels_exa)
        out_log.write('{}\n'.format(exp_str))
        out_log.flush()

        if mem_map is not None:
            n_cols = 12
            mem_map = mem_map.reshape(-1, n_cols)
            print('mm', mem_map.shape)
            config_array = numpy.array([args.exp_id,
                                        train_avg_ll, valid_avg_ll, test_avg_ll,
                                        marg_train_avg_ll, marg_valid_avg_ll, marg_test_avg_ll,
                                        mpe_class_accuracy, map_class_accuracy,
                                        mpe_labels_jac, mpe_labels_ham, mpe_labels_exa,
                                        ])
            print('config', config_array.shape)
            mem_map[args.exp_id, :] = config_array
            print('mm', mem_map.shape)
            mem_map.flush()
            logging.info('Memmap {} flushed'.format(args.memmap))


if args.marginal_ll_emb:
    print('computing marginal log likelihood for embeddings')
    # marg_train = train[:, :n_emb_features]
    # marg_valid = valid[:, :n_emb_features]
    # marg_test = test[:, :n_emb_features]

    marg_out_rvs = {i for i in range(n_emb_features,   test.shape[1])}
    print('RVs to marginalize {}'.format(marg_out_rvs))
    # label_mpe_data = numpy.copy(test[:, n_emb_features + 1:])
    # label_mpe_data[:, n_emb_features:] = numpy.nan
    marg_spn = spn.root.marginalizeOut(marginals=marg_out_rvs)

    eval_start_t = perf_counter()
    marg_train_lls = marg_spn.eval(train)
    marg_valid_lls = marg_spn.eval(valid)
    marg_test_lls = marg_spn.eval(test)
    eval_end_t = perf_counter()
    eval_time = eval_end_t - eval_start_t
    logging.info('\n\n*****Spn eval in {} secs! *****'.format(eval_time))

    marg_train_avg_ll = numpy.mean(marg_train_lls)
    marg_valid_avg_ll = numpy.mean(marg_valid_lls)
    marg_test_avg_ll = numpy.mean(marg_test_lls)
    logging.info("\tmarg train:\t{}\t(min:{}\tmax:{})".format(marg_train_avg_ll,
                                                              marg_train_lls.min(), marg_train_lls.max()))
    logging.info("\tmarg valid:\t{}\t(min:{}\tmax:{})".format(marg_valid_avg_ll,
                                                              marg_valid_lls.min(), marg_valid_lls.max()))
    logging.info("\tmarg test:\t{}\t(min:{}\tmax:{})".format(marg_test_avg_ll,
                                                             marg_test_lls.min(), marg_test_lls.max()))


if args.masks:

    mask_emb_features = args.mask_emb_features
    assert len(mask_emb_features) == len(args.masks)
    assert sum(mask_emb_features) == args.emb_features

    cum_mask_emb_features = numpy.cumsum([0] + mask_emb_features)

    n_top_images = args.n_images
    n_square = int(numpy.sqrt(n_top_images))
    n_rows, n_cols = None, None
    if args.n_rows is not None and args.n_cols is not None:
        n_rows, n_cols = args.n_rows, args.n_cols
    else:
        n_rows, n_cols = n_square, n_square

    fig_width = 3
    fig_height = 12
    fig_size = (fig_width, fig_height)

    for i, m in enumerate(args.masks):

        masked_dec_path = os.path.join(args.ae_path, m, 'ae.raelk.mnist.decoder.0')
        masked_decoder = load_ae_decoder(masked_dec_path)

        #
        # We want to predict some masked parts given all the rest
        # argmax p(M_{i}| C, L, M_{0}, ..., M_{i-1}, M_{i+1}, ..., M_{k})
        logging.info('\n\n*** Computing MPE predictions for mask {} ******'.format(m))
        mask_mpe_data = numpy.copy(test)
        logging.info("Labeled data (first example)\n{}".format(mask_mpe_data[0]))

        query_ids = numpy.arange(cum_mask_emb_features[i], cum_mask_emb_features[i + 1])
        logging.debug('\tquery ids {}'.format(query_ids))
        mask_mpe_data[:, query_ids] = numpy.nan
        logging.info("Masked label data (first example)\n{}".format(mask_mpe_data[0]))

        mask_mpe_probs, mask_mpe_res = spn.root.mpe_eval(mask_mpe_data)
        logging.info("MPE prediction (first example)\n{}".format(mask_mpe_res[0]))

        #
        # then we have to decode it
        masked_emb = mask_mpe_res[:, query_ids]
        dec_img_samples = decode_predictions(masked_emb, masked_decoder)

        orig_masked_emb = test[:, query_ids]
        dec_test_emb = decode_predictions(orig_masked_emb, masked_decoder)

        #
        # combining to original images
        test_rec = numpy.copy(orig_test_x)
        mask_ids = get_image_mask(orig_test_x.shape[1], n_cols=28, mask=m)
        test_rec[:, mask_ids] = dec_img_samples

        #
        # visualize and save them
        mask_img_size = MASKED_IMAGE_SIZE[m]
        decoded_mask_path = os.path.join(args.output, 'dec.{}.{}'.format(m, n_top_images))
        plot_digits_matrix(test_rec[:n_top_images], m=n_rows, n=n_cols,
                           # img_size=mask_img_size,
                           fig_size=fig_size,
                           w_space=0.0,
                           h_space=0.0,
                           masking=mask_ids,
                           mask_cmap=RED_CMAP,
                           output=decoded_mask_path, show=args.show_images)
        logging.info('Dumped vis for first {} images for mask {} to:\n\t{}'.format(n_top_images,
                                                                                   m,
                                                                                   decoded_mask_path))

        #
        # combining autoencoder to original images
        auto_test_rec = numpy.copy(orig_test_x)
        # mask_ids = get_image_mask(orig_test_x.shape[1], n_cols=28, mask=m)
        auto_test_rec[:, mask_ids] = dec_test_emb

        #
        # visualize and save them
        mask_img_size = MASKED_IMAGE_SIZE[m]
        auto_decoded_mask_path = os.path.join(
            args.output, 'auto.dec.{}.{}'.format(m, n_top_images))
        plot_digits_matrix(auto_test_rec[:n_top_images], m=n_rows, n=n_cols,
                           # img_size=mask_img_size,
                           fig_size=fig_size,
                           w_space=0.0,
                           h_space=0.0,
                           masking=mask_ids,
                           mask_cmap=BLUE_CMAP,
                           output=auto_decoded_mask_path, show=args.show_images)
        logging.info('Dumped vis for first {} images for masked auto {} to:\n\t{}'.format(n_top_images,
                                                                                          m,
                                                                                          auto_decoded_mask_path))

        #
        # visualize original images as well
        orig_decoded_mask_path = os.path.join(args.output, 'orig.{}.{}'.format(m, n_top_images))
        plot_digits_matrix(orig_test_x[:n_top_images], m=n_rows, n=n_cols,
                           fig_size=fig_size,
                           w_space=0.0,
                           h_space=0.0,
                           output=orig_decoded_mask_path, show=args.show_images)
        logging.info('Dumped vis for first {} images for original images to:\n\t{}'.format(n_top_images,
                                                                                           orig_decoded_mask_path))

    test_labels = test[:, n_emb_features].astype(numpy.int32)
    print(test_labels[n_top_images])


if args.mpe_labels:

    label_mpe_data = None
    label_mpe_probs = None
    label_mpe_res = None

    if args.marginalize_labels:
        print('Marginalizing labels out (class MPE prediction only)')

        marg_out_rvs = {i for i in range(n_emb_features + 1,   test.shape[1])}
        print('RVs to marginalize {}'.format(marg_out_rvs))
        # label_mpe_data = numpy.copy(test[:, n_emb_features + 1:])
        # label_mpe_data[:, n_emb_features:] = numpy.nan
        label_mpe_data = numpy.copy(test)
        label_mpe_data[:, n_emb_features:] = numpy.nan
        marg_spn = spn.root.marginalizeOut(marginals=marg_out_rvs)
        label_mpe_probs, label_mpe_res = marg_spn.mpe_eval(label_mpe_data)

        test_labels = test[:, n_emb_features].astype(numpy.int32)
        mpe_labels = label_mpe_res[:, n_emb_features].astype(numpy.int32)
        print(test_labels[:20], mpe_labels[:20])
        print('class accuracy {}'.format(compute_scores(test_labels, mpe_labels, 'accuracy')))

    else:
        label_mpe_data = numpy.copy(test)
        label_mpe_data[:, n_emb_features:] = numpy.nan
        # mpe_data = mpe_data[[0, 9, 3, 4], :]
        print("Masked label data\n", label_mpe_data[:n_emb_features])

        label_mpe_probs, label_mpe_res = spn.root.mpe_eval(label_mpe_data)
        # mpe_probs, mpe_res = spn.root.mpe_eval(samples)
        print("label MPE\n", label_mpe_res[:n_emb_features])
        numpy.save('label-mpe-ass', label_mpe_res)

        if args.mpe_class:

            print('MPE for class + other labels jointly (no marginalization)')
            #
            # class attribute
            test_labels = test[:, n_emb_features].astype(numpy.int32)
            mpe_labels = label_mpe_res[:, n_emb_features].astype(numpy.int32)
            print(test_labels[:20], mpe_labels[:20])
            print('class accuracy {}'.format(compute_scores(test_labels, mpe_labels, 'accuracy')))

            print('\nother labels')
            test_labels = test[:, n_emb_features + 1:].astype(numpy.int32)
            mpe_labels = label_mpe_res[:, n_emb_features + 1:].astype(numpy.int32)
            print('accuracy {}'.format(compute_scores(test_labels, mpe_labels, 'accuracy')))
            print('exact {}'.format(compute_scores(test_labels, mpe_labels, 'exact')))
            print('hamming {}'.format(compute_scores(test_labels, mpe_labels, 'hamming')))
            print('jaccard {}'.format(compute_scores(test_labels, mpe_labels, 'jaccard')))
        else:
            test_labels = test[:, n_emb_features:].astype(numpy.int32)
            mpe_labels = label_mpe_res[:, n_emb_features:].astype(numpy.int32)
            print('accuracy {}'.format(compute_scores(test_labels, mpe_labels, 'accuracy')))
            print('exact {}'.format(compute_scores(test_labels, mpe_labels, 'exact')))
            print('hamming {}'.format(compute_scores(test_labels, mpe_labels, 'hamming')))
            print('jaccard {}'.format(compute_scores(test_labels, mpe_labels, 'jaccard')))

            #
            # only for zero class
            # zeroclass = numpy.array([1., 0., 0., 0., 1., 0.])
            # zero_samples = []
            # for l in test_labels:
            #     if numpy.all(numpy.isclose(l, zeroclass)):
            #         zero_samples.append(True)
            #     else:
            #         zero_samples.append(False)
            # zero_samples = numpy.array(zero_samples, dtype=bool)

            #
            # only for one class
            class_n = 4
            class_sample_ids = orig_test_y == class_n
            print('for {} classed samples (class {})'.format(class_sample_ids.sum(), class_n))
            print('exact {}'.format(compute_scores(test_labels[class_sample_ids, :],
                                                   mpe_labels[class_sample_ids, :], 'exact')))
            print('hamming {}'.format(compute_scores(test_labels[class_sample_ids, :],
                                                     mpe_labels[class_sample_ids, :], 'hamming')))
            print('jaccard {}'.format(compute_scores(test_labels[class_sample_ids, :],
                                                     mpe_labels[class_sample_ids, :], 'jaccard')))

            print('some test zero samples {}'.format(test_labels[class_sample_ids, :][:20]))
            print('some zero sample predictions {}'.format(mpe_labels[class_sample_ids, :][:20]))

            # exact_zero_predictions = []
            # for l in mpe_labels:
            #     if numpy.all(numpy.isclose(l, zeroclass.astype(numpy.int32))):
            #         exact_zero_predictions.append(True)
            #     else:
            #         exact_zero_predictions.append(False)
            # exact_zero_predictions = numpy.array(exact_zero_predictions, dtype=bool)
            # print('# exact zero predictions', sum(exact_zero_predictions))
            # print('some exact zero sample predictions {}'.format(
            #     mpe_labels[exact_zero_predictions, :][:20]))

            class_sample = numpy.array([1., 0., 0., 0., 1., 0.])
            exact_class_predictions = []
            for l in mpe_labels:
                if numpy.all(numpy.isclose(l, class_sample.astype(numpy.int32))):
                    exact_class_predictions.append(True)
                else:
                    exact_class_predictions.append(False)
            exact_class_predictions = numpy.array(exact_class_predictions, dtype=bool)
            print('# exact zero predictions', sum(exact_class_predictions))
            print('some exact zero sample predictions {}'.format(
                mpe_labels[exact_class_predictions, :][:20]))


worlds = None
if args.all_worlds:
    logging.info('Generating all binary configurations for {} labels'.format(n_aug_features))

    worlds = [i for i in itertools.product(range(2), repeat=n_aug_features)]

    mpe_data = numpy.array([numpy.array([numpy.nan] * n_emb_features + list(w))
                            for w in worlds])
    logging.info('There are {} configurations'.format(len(mpe_data)))

else:
    n_train_samples = 10
    mpe_data = numpy.copy(train[:n_train_samples])
    # print("train data\n", mpe_data)
    mpe_data[:, :n_emb_features] = numpy.nan
    # mpe_data = mpe_data[[0, 9, 3, 4], :]
    # print("Masked data\n", mpe_data)

    # one_more_row = numpy.array([numpy.nan] * n_emb_features + [1, 1, 0, 0, 0])
    # one_more_row = one_more_row.reshape(-1, one_more_row.shape[0])
    # # print(one_more_row)
    # mpe_data = numpy.concatenate([mpe_data,
    #                               one_more_row], axis=0)

    worlds = [w for w in mpe_data[:, n_emb_features:]]


# dec_path = args.decoder
dec_path = os.path.join(args.ae_path, 'ae.raelk.mnist.decoder.0')
decoder = load_ae_decoder(dec_path)
print('loaded the decoder from {}'.format(dec_path))
print(decoder)

# worlds = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
#           [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1]]

if args.samples:

    all_samples = []
    all_embeddings = []
    n_samples = args.samples
    for i, w in enumerate(worlds):
        #
        # create sample matrix to evaluate
        sample_query_data = numpy.array([[numpy.nan] * n_emb_features + list(w)
                                         for i in range(n_samples)])
        print('sample query', sample_query_data)
        probs, samples = spn.root.sample(sample_query_data, rand_gen=rand_gen)
        print(samples)
        #
        # filtering samples by probability
        if args.filter_samples_prob:
            samples_lls = spn.root.eval(samples)
            samples_inv_ids = numpy.argsort(samples_lls)
            samples_ids = samples_inv_ids[::-1]
            samples = samples[samples_ids[:args.filter_samples_prob], :]

        emb_samples = samples[:, :n_emb_features]
        print(i)
        print(emb_samples)
        print(probs)
        all_embeddings.append(emb_samples)

        #
        # decoding them
        dec_img_samples = decode_predictions(emb_samples, decoder)
        dec_samples_path = os.path.join(args.output, 'dec-cond-samples')
        numpy.save(dec_samples_path, dec_img_samples)

        all_samples.append(dec_img_samples)
        # for j, img in enumerate(dec_img_samples):
        #     print(j)
        #     plot_digit(img)
        samples_out_path = None
        if args.output:
            w_str = '.'.join(str(int(wi)) for wi in w)
            print(w_str)
            samples_out_path = os.path.join(args.output, 'cond-samples-{}-{}'.format(i + 1, w_str))
        n_square = int(numpy.sqrt(len(dec_img_samples)))
        plot_digits_matrix(dec_img_samples, m=n_square, n=n_square,
                           w_space=0.0,
                           h_space=0.0,
                           output=samples_out_path, show=args.show_images)

    # samples = numpy.array([numpy.array([numpy.nan] * 10 + [0, 0, 1, 0, 1]),
    #                        numpy.array([numpy.nan] * 10 + [0, 0, 1, 0, 1]),
    #                        numpy.array([numpy.nan] * 10 + [0, 0, 1, 0, 1]), ])
    # print('synth samples {}'.format(samples))

    if args.vis_2d:

        all_sample_images = numpy.concatenate(all_samples, axis=0)
        all_sample_embeddings = numpy.concatenate(all_embeddings, axis=0)

        # y_labels = binary_labels_to_int(worlds)
        dup_worlds = []
        for w_i in worlds:
            for k in range(n_samples):
                dup_worlds.append(w_i)
        y_labels = binary_labels_to_int(dup_worlds)

        print(all_sample_embeddings)
        twod_embeddings = None
        if all_sample_embeddings.shape[1] == 2:
            #
            # transforming embedding space
            pos_mpe_embeddings = numpy.copy(all_sample_embeddings)
            pos_mpe_embeddings[:, 0] = pos_mpe_embeddings[:, 0] + \
                numpy.abs(pos_mpe_embeddings[:, 0].min())
            pos_mpe_embeddings[:, 1] = pos_mpe_embeddings[:, 1] + \
                numpy.abs(pos_mpe_embeddings[:, 1].min())
            log_pos_mpe_embeddings = numpy.log(pos_mpe_embeddings + 1)
            twod_embeddings = log_pos_mpe_embeddings
        else:
            print('Projecting {} dimensions to 2 with tsne'.format(all_sample_embeddings.shape[1]))
            t0 = perf_counter()
            perplexity = 30
            n_iter = 1000
            transformer = manifold.TSNE(n_components=2,
                                        # init=args.init,
                                        perplexity=perplexity,
                                        # early_exaggeration=args.early_exaggeration,
                                        # learning_rate=args.learning_rate,
                                        n_iter=n_iter,
                                        # n_iter_without_progress=args.n_iter_without_progress,
                                        # metric=args.metric,
                                        # method=args.method,
                                        # angle=angle,
                                        random_state=rand_gen,
                                        verbose=args.verbose)
            # transformer = manifold.MDS(n_components=2)
            twod_embeddings = transformer.fit_transform(all_sample_embeddings)
            t1 = perf_counter()
            logging.info("t-SNE computed in %.3f sec" % (t1 - t0))

        print(twod_embeddings)
        plot_embedding(all_sample_images, y_labels, twod_embeddings, '2d-emb', min_dist=20.0,
                       marker='x',
                       marker_size=100,
                       marker_width=2,
                       fig_size=(16, 16),
                       img_size=(28, 28),
                       max_instances=5000,
                       zoom=0.5,
                       rand_gen=rand_gen,
                       output=None,
                       invert=None,
                       show=True)

output_path = None


if args.mpe_embeddings:

    mpe_probs, mpe_res = spn.root.mpe_eval(mpe_data)
    # mpe_probs, mpe_res = spn.root.mpe_eval(samples)
    print("MPE\n", mpe_res)
    numpy.save('mpe-ass', mpe_res)

    mpe_embeddings = mpe_res[:, :n_emb_features]
    print(mpe_embeddings.shape)

    # for t in train:
    #     for m in mpe_embeddings:
    #         if numpy.all(numpy.isclose(m[:10], t[:10])):
    #             print('close', m, t)
    #         for i in range(10):
    #             if numpy.isclose(m[i], t[i]):
    #                 print('close', m[i], t[i])

    dec_imgs = decode_predictions(mpe_embeddings, decoder)
    # print('decoded samples {}\n{}'.format(dec_imgs.shape, dec_imgs))

    nn_imgs = get_nearest_neighbour(dec_imgs, orig_train_x)

    for i, (img, w) in enumerate(zip(dec_imgs, worlds)):
        # print(mpe_res[i])

        logging.info('considering world {}'.format(w))
        w_str = '.'.join(str(int(wi)) for wi in w)

        if args.output:
            output_path = os.path.join(args.output, 'mpe-{}-{}'.format(i + 1, w_str))

        plot_digit(img, output=output_path, show=args.show_images)

        nn_id, nn_img = nn_imgs[i]
        if args.output:
            output_path = os.path.join(args.output, 'nn-sample-{}-{}'.format(i + 1, w_str))
        # print(nn_img)
        plot_digit(nn_img, output=output_path, show=args.show_images)

    if args.vis_2d:
        y_labels = binary_labels_to_int(worlds)
        #
        # transforming embedding space
        print(mpe_embeddings)
        pos_mpe_embeddings = numpy.copy(mpe_embeddings)
        pos_mpe_embeddings[:, 0] = pos_mpe_embeddings[:, 0] + \
            numpy.abs(pos_mpe_embeddings[:, 0].min())
        pos_mpe_embeddings[:, 1] = pos_mpe_embeddings[:, 1] + \
            numpy.abs(pos_mpe_embeddings[:, 1].min())

        log_pos_mpe_embeddings = numpy.log(pos_mpe_embeddings + 1)
        print(log_pos_mpe_embeddings)
        plot_embedding(dec_imgs, y_labels, log_pos_mpe_embeddings, '2d-emb', min_dist=0.0,
                       marker='x',
                       marker_size=100,
                       marker_width=2,
                       fig_size=(16, 16),
                       img_size=(28, 28),
                       max_instances=5000,
                       zoom=1.0,
                       rand_gen=rand_gen,
                       output=None,
                       invert=None,
                       show=True)


if args.combine:

    print('combining')
    combined_image = (orig_train_x[0] + orig_train_x[1]) / 2
    plot_digit(orig_train_x[0])
    plot_digit(orig_train_x[1])
    plot_digit(combined_image)

    comb_images = numpy.array([combined_image])

    enc_path = dec_path = os.path.join(args.ae_path, 'ae.raelk.mnist.encoder.0')
    encoder = load_ae_encoder(enc_path)
    print('loaded the encoder from {}'.format(enc_path))
    print(encoder)
    enc_imgs = encode_predictions(comb_images, encoder)

    enc_mpe_data = numpy.concatenate([enc_imgs[0], numpy.array([numpy.nan] * n_aug_features)])
    enc_mpe_data = enc_mpe_data.reshape(-1, enc_mpe_data.shape[0])
    enc_mpe_probs, enc_mpe_res = spn.root.mpe_eval(enc_mpe_data)
    print(enc_mpe_res)
