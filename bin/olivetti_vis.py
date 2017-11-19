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
from bin.merge_olivetti_raelk import plot_digit
from bin.merge_olivetti_raelk import plot_digits_matrix
from bin.merge_olivetti_raelk import load_train_val_test_splits
from experiments.mnist.augmenting_mnist import get_image_mask
from bin.merge_olivetti_raelk import RED_CMAP, BLUE_CMAP

from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import average_precision_score
from sklearn import manifold

from keras.models import load_model


def left_mask(n_features, n_cols):
    feature_ids = numpy.arange(n_features)
    half_cols = n_cols // 2
    return (feature_ids % n_cols) < half_cols


def right_mask(n_features, n_cols):
    feature_ids = numpy.arange(n_features)
    half_cols = n_cols // 2
    return (feature_ids % n_cols) >= half_cols


def upper_mask(n_features):
    feature_ids = numpy.zeros(n_features, dtype=bool)
    feature_ids[:n_features // 2] = True
    return feature_ids


def lower_mask(n_features):
    feature_ids = numpy.zeros(n_features, dtype=bool)
    feature_ids[n_features // 2:] = True
    return feature_ids


def get_image_mask(n_features, n_cols=64, mask=None):
    feature_mask = None

    if mask == 'left':
        feature_mask = left_mask(n_features, n_cols)
    elif mask == 'right':
        feature_mask = right_mask(n_features, n_cols)
    elif mask == 'up':
        feature_mask = upper_mask(n_features)
    elif mask == 'down':
        feature_mask = lower_mask(n_features)

    return feature_mask


def decode_predictions(repr_preds, ae_decoder):

    preds = ae_decoder.predict(repr_preds)
    n_instances = preds.shape[0]
    preds = preds.reshape(n_instances, numpy.prod(preds.shape[1:]))
    return preds


def encode_predictions(imgs, ae_encoder):

    reprs = ae_encoder.predict(imgs)
    return reprs


def load_ae_decoder(model_path):

    logging.info('Loading AE model from {}'.format(model_path))

    #
    # loading with keras
    load_start_t = perf_counter()
    ae_decoder = load_model(model_path)
    load_end_t = perf_counter()
    logging.info('\tdone in {}'.format(load_end_t - load_start_t))

    return ae_decoder


def load_ae_encoder(model_path):

    logging.info('Loading AE model from {}'.format(model_path))

    #
    # loading with keras
    load_start_t = perf_counter()
    ae_encoder = load_model(model_path)
    load_end_t = perf_counter()
    logging.info('\tdone in {}'.format(load_end_t - load_start_t))

    return ae_encoder


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
                    default='exp/olivetti-ae/lr/',
                    help='Specify dataset dir (default data/)')

parser.add_argument('--orig-data', type=str,
                    default='olivetti.pklz',
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
                    default=None,
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

# parser.add_argument('--aug-features', type=int, nargs='?',
#                     default=5,
#                     help='Number of augmented features')

parser.add_argument('--emb-features', type=int, nargs='?',
                    default=10,
                    help='Number of embedded features')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')

# parser.add_argument('--all-worlds', action='store_true',
#                     help='Compute MPE and visualize for all possible label combinations')

# parser.add_argument('--mpe-labels', action='store_true',
#                     help='Predicting labels for test data')

parser.add_argument('--combine', action='store_true',
                    help='Combine image')

# parser.add_argument('--vis-2d', action='store_true',
#                     help='visualize embedding as 2d images')

parser.add_argument('--show-images', action='store_true',
                    help='Show single images')

# parser.add_argument('--mpe-embeddings', action='store_true',
#                     help='Show MPE predictions for embeddings')

# parser.add_argument('--mpe-class', action='store_true',
#                     help='Class variable is present')

# parser.add_argument('--marginalize-labels', action='store_true',
#                     help='Marginalizing labels, class variable is present')

# parser.add_argument('--privileged-inference', action='store_true',
#                     help='All inference results for privileged case')

# parser.add_argument('--filter-samples-prob', type=int, nargs='?',
#                     default=None,
#                     help='All inference results for privileged case')

# parser.add_argument('--marginal-ll-emb', action='store_true',
#                     help='Compute the marginal log likelihood on the embedding features only')

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
                                              'olivetti',
                                              train_ext=None,
                                              valid_ext=None,
                                              test_ext=None,
                                              y_only=False,
                                              dtype=numpy.float64)

print(len(orig_fold_splits))
print(len(orig_fold_splits[0]))
(orig_train_x, orig_train_y), (orig_valid_x,
                               orig_valid_y), (orig_test_x, orig_test_y) = orig_fold_splits[0]
print('Orig split sizes', orig_train_x.shape, orig_valid_x.shape, orig_test_x.shape)

# orig_train_x, orig_train_y = orig_train
# orig_valid_x, orig_valid_y = orig_valid
# orig_test_x, orig_test_y = orig_test


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

MASKED_IMAGE_SIZE = {'left': (64, 32),
                     'right': (64, 32),
                     'up': (64, 32),
                     'down': (64, 32)}


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

        masked_dec_path = os.path.join(args.ae_path, m, 'ae.raelk.olivetti.decoder.0')
        masked_decoder = load_ae_decoder(masked_dec_path)

        for orig_split, split_name, ae_split in zip([orig_train_x, orig_valid_x, orig_test_x],
                                                    ['train', 'valid', 'test'],
                                                    [train, valid, test],
                                                    ):
            #
            # We want to predict some masked parts given all the rest
            # argmax p(M_{i}| C, L, M_{0}, ..., M_{i-1}, M_{i+1}, ..., M_{k})
            logging.info('\n\n*** Computing MPE predictions for mask {} ******'.format(m))
            mask_mpe_data = numpy.copy(ae_split)
            logging.info("Labeled data (first example)\n{}".format(mask_mpe_data[0]))

            query_ids = numpy.arange(cum_mask_emb_features[i], cum_mask_emb_features[i + 1])
            logging.debug('\tquery ids {}'.format(query_ids))
            mask_mpe_data[:, query_ids] = numpy.nan
            logging.info("Masked label data (first example)\n{}".format(mask_mpe_data[0]))

            mask_mpe_probs, mask_mpe_res = spn.root.mpe_eval(mask_mpe_data)
            logging.info('MPE pred shape {}'.format(mask_mpe_res.shape))
            logging.info("MPE prediction (first example)\n{}".format(mask_mpe_res[0]))
            logging.info("MPE prediction (second example)\n{}".format(mask_mpe_res[1]))
            logging.info("MPE prediction (third example)\n{}".format(mask_mpe_res[2]))

            #
            # then we have to decode it
            masked_emb = mask_mpe_res[:, query_ids]
            logging.info('mask emb shape {} {}'.format(masked_emb.shape, query_ids.shape))
            dec_img_samples = decode_predictions(masked_emb, masked_decoder)
            logging.info('dec img samples shape {}'.format(dec_img_samples.shape))

            orig_masked_emb = ae_split[:, query_ids]
            dec_test_emb = decode_predictions(orig_masked_emb, masked_decoder)

            #
            # combining to original images
            test_rec = numpy.copy(orig_split)
            mask_ids = get_image_mask(orig_split.shape[1], n_cols=64, mask=m)
            test_rec[:, mask_ids] = dec_img_samples

            #
            # visualize and save them
            mask_img_size = MASKED_IMAGE_SIZE[m]
            decoded_mask_path = os.path.join(args.output,
                                             'dec.{}.{}.{}'.format(m, n_top_images, split_name))
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
            auto_test_rec = numpy.copy(orig_split)
            # mask_ids = get_image_mask(orig_test_x.shape[1], n_cols=28, mask=m)
            auto_test_rec[:, mask_ids] = dec_test_emb

            #
            # visualize and save them
            mask_img_size = MASKED_IMAGE_SIZE[m]
            auto_decoded_mask_path = os.path.join(
                args.output, 'auto.dec.{}.{}.{}'.format(m, n_top_images, split_name))
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
            orig_decoded_mask_path = os.path.join(
                args.output, 'orig.{}.{}.{}'.format(m, n_top_images, split_name))
            plot_digits_matrix(orig_split[:n_top_images], m=n_rows, n=n_cols,
                               fig_size=fig_size,
                               w_space=0.0,
                               h_space=0.0,
                               output=orig_decoded_mask_path, show=args.show_images)
            logging.info('Dumped vis for first {} images for original images to:\n\t{}'.format(n_top_images,
                                                                                               orig_decoded_mask_path))
        # #
        # # We want to predict some masked parts given all the rest
        # # argmax p(M_{i}| C, L, M_{0}, ..., M_{i-1}, M_{i+1}, ..., M_{k})
        # logging.info('\n\n*** Computing MPE predictions for mask {} ******'.format(m))
        # mask_mpe_data = numpy.copy(test)
        # logging.info("Labeled data (first example)\n{}".format(mask_mpe_data[0]))

        # query_ids = numpy.arange(cum_mask_emb_features[i], cum_mask_emb_features[i + 1])
        # logging.debug('\tquery ids {}'.format(query_ids))
        # mask_mpe_data[:, query_ids] = numpy.nan
        # logging.info("Masked label data (first example)\n{}".format(mask_mpe_data[0]))

        # mask_mpe_probs, mask_mpe_res = spn.root.mpe_eval(mask_mpe_data)
        # logging.info('MPE pred shape {}'.format(mask_mpe_res.shape))
        # logging.info("MPE prediction (first example)\n{}".format(mask_mpe_res[0]))
        # logging.info("MPE prediction (second example)\n{}".format(mask_mpe_res[1]))
        # logging.info("MPE prediction (third example)\n{}".format(mask_mpe_res[2]))

        # #
        # # then we have to decode it
        # masked_emb = mask_mpe_res[:, query_ids]
        # logging.info('mask emb shape {} {}'.format(masked_emb.shape, query_ids.shape))
        # dec_img_samples = decode_predictions(masked_emb, masked_decoder)
        # logging.info('dec img samples shape {}'.format(dec_img_samples.shape))

        # orig_masked_emb = test[:, query_ids]
        # dec_test_emb = decode_predictions(orig_masked_emb, masked_decoder)

        # #
        # # combining to original images
        # test_rec = numpy.copy(orig_test_x)
        # mask_ids = get_image_mask(orig_test_x.shape[1], n_cols=64, mask=m)
        # test_rec[:, mask_ids] = dec_img_samples

        # #
        # # visualize and save them
        # mask_img_size = MASKED_IMAGE_SIZE[m]
        # decoded_mask_path = os.path.join(args.output, 'dec.{}.{}'.format(m, n_top_images))
        # plot_digits_matrix(test_rec[:n_top_images], m=n_rows, n=n_cols,
        #                    # img_size=mask_img_size,
        #                    fig_size=fig_size,
        #                    w_space=0.0,
        #                    h_space=0.0,
        #                    masking=mask_ids,
        #                    mask_cmap=RED_CMAP,
        #                    output=decoded_mask_path, show=args.show_images)
        # logging.info('Dumped vis for first {} images for mask {} to:\n\t{}'.format(n_top_images,
        #                                                                            m,
        #                                                                            decoded_mask_path))

        # #
        # # combining autoencoder to original images
        # auto_test_rec = numpy.copy(orig_test_x)
        # # mask_ids = get_image_mask(orig_test_x.shape[1], n_cols=28, mask=m)
        # auto_test_rec[:, mask_ids] = dec_test_emb

        # #
        # # visualize and save them
        # mask_img_size = MASKED_IMAGE_SIZE[m]
        # auto_decoded_mask_path = os.path.join(
        #     args.output, 'auto.dec.{}.{}'.format(m, n_top_images))
        # plot_digits_matrix(auto_test_rec[:n_top_images], m=n_rows, n=n_cols,
        #                    # img_size=mask_img_size,
        #                    fig_size=fig_size,
        #                    w_space=0.0,
        #                    h_space=0.0,
        #                    masking=mask_ids,
        #                    mask_cmap=BLUE_CMAP,
        #                    output=auto_decoded_mask_path, show=args.show_images)
        # logging.info('Dumped vis for first {} images for masked auto {} to:\n\t{}'.format(n_top_images,
        #                                                                                   m,
        #                                                                                   auto_decoded_mask_path))

        # #
        # # visualize original images as well
        # orig_decoded_mask_path = os.path.join(args.output, 'orig.{}.{}'.format(m, n_top_images))
        # plot_digits_matrix(orig_test_x[:n_top_images], m=n_rows, n=n_cols,
        #                    fig_size=fig_size,
        #                    w_space=0.0,
        #                    h_space=0.0,
        #                    output=orig_decoded_mask_path, show=args.show_images)
        # logging.info('Dumped vis for first {} images for original images to:\n\t{}'.format(n_top_images,
        #                                                                                    orig_decoded_mask_path))

    # test_labels = test[:, n_emb_features].astype(numpy.int32)
    # print(test_labels[n_top_images])
