import os
import gzip
import logging
import pickle

import numpy
from experiments.mnist.augmenting_mnist import plot_digits_matrix

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit


OUTPUT_PATH = 'experiments/olivetti/'
DISCARD_HEADER = True


IMG_SIZE = (64, 64)

TEST_PROPORTION = 0.2
VALID_PROPORTION = 0.2

RAND_SEED = 1337

if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    rand_gen = numpy.random.RandomState(RAND_SEED)

    olivetti = fetch_olivetti_faces()
    data = olivetti.data
    target = olivetti.target
    logging.info('Loaded Olivetti data x:{} y:{}'.format(data.shape, target.shape))

    #
    # making three splits
    train_x, train_y = None, None
    valid_x, valid_y = None, None
    test_x, test_y = None, None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_PROPORTION, random_state=rand_gen)
    for train_index, test_index in sss.split(data, target):
        train_x, train_y = data[train_index], target[train_index]
        test_x, test_y = data[test_index], target[test_index]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALID_PROPORTION, random_state=rand_gen)
    for train_index, valid_index in sss.split(train_x, train_y):
        valid_x, valid_y = train_x[valid_index], train_y[valid_index]
        train_x, train_y = train_x[train_index], train_y[train_index]

    logging.info('\Split into')
    logging.info('\n\ttrain:\t{}\t{}\n\tvalid:\t{}\t{}\n\ttest:\t{}\t{}'.format(train_x.shape,
                                                                                train_y.shape,
                                                                                valid_x.shape,
                                                                                valid_y.shape,
                                                                                test_x.shape,
                                                                                test_y.shape))

    #
    #

    #
    # visualizing samples
    n_top_images = 64
    n_square = int(numpy.sqrt(n_top_images))
    output_path = os.path.join(OUTPUT_PATH, 'train.samples')
    plot_digits_matrix(train_x[:n_top_images],
                       m=n_square, n=n_square,
                       img_size=IMG_SIZE,
                       w_space=0.0,
                       h_space=0.0,
                       output=output_path,
                       show=True)

    output_path = os.path.join(OUTPUT_PATH, 'valid.samples')
    plot_digits_matrix(valid_x[:n_top_images],
                       m=n_square, n=n_square,
                       img_size=IMG_SIZE,
                       w_space=0.0,
                       h_space=0.0,
                       output=output_path,
                       show=True)

    output_path = os.path.join(OUTPUT_PATH, 'test.samples')
    plot_digits_matrix(test_x[:n_top_images],
                       m=n_square, n=n_square,
                       img_size=IMG_SIZE,
                       w_space=0.0,
                       h_space=0.0,
                       output=output_path,
                       show=True)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    repr_save_path = os.path.join(OUTPUT_PATH, 'olivetti.pklz')
    fold = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    with gzip.open(repr_save_path, 'wb') as f:
        print('Saving splits to {}'.format(repr_save_path))
        pickle.dump(fold, f, protocol=4)
        print('Saved.')
