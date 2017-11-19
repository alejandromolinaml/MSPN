import sys
import os
import logging
from time import perf_counter
import argparse

import numpy
from sklearn.model_selection import StratifiedShuffleSplit

from mlutils.datasets import remove_missing_value_samples
from mlutils.datasets import load_feature_info_preprocess
from mlutils.datasets import load_uci_data_from_text
from mlutils.datasets import estimate_categorical_domain
from mlutils.datasets import estimate_continuous_domain
from mlutils.datasets import estimate_continuous_domain_range
from mlutils.datasets import estimate_continuous_domain_min_max
from mlutils.datasets import save_feature_info_dict
from mlutils.datasets import uci_data_to_numpy
from mlutils.datasets import get_feature_formats

TRAIN_DATA_EXT = 'train'
VALID_DATA_EXT = 'valid'
TEST_DATA_EXT = 'test'
MLC_TRAIN_EXT = 'data'
MLC_TEST_EXT = 'test'
MISS_VAL_STR = '?'
HEADER = 1
FEATURE_INFO_EXT = 'features'
NON_MISS_STR = 'nomiss'
PROC_STR = 'proc'

if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()

    parser.add_argument("datadir", type=str,
                        help='Dataset directory path')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./datasets/autism/proc',
                        help='Output dir path')

    parser.add_argument('-b', '--bins', type=str, nargs='?',
                        default='blocks',
                        help='Binning method (blocks|unique|auto|fd|...)')

    parser.add_argument('--valid-perc', type=float,
                        default=0.1,
                        help='Percentage of training set to reserve for validation')

    parser.add_argument('--test-perc', type=float,
                        default=0.3,
                        help='Percentage of training set to reserve for testing')

    parser.add_argument('--miss-val', type=str,
                        default=MISS_VAL_STR,
                        help='Missing value string representation')

    parser.add_argument('--header', type=int,
                        default=HEADER,
                        help='Header length')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=1337,
                        help='Random generator seed')

    parser.add_argument('--target', type=int, nargs='?',
                        default=26,
                        help='target feature id (default 26)')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    # parsing the args
    args = parser.parse_args()

    rand_gen = numpy.random.RandomState(args.seed)

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    os.makedirs(args.output, exist_ok=True)
    bin_method = args.bins

    data_name = 'autism.{}'.format(MLC_TRAIN_EXT)
    data_path = os.path.join(args.datadir, data_name)
    data = numpy.loadtxt(data_path, delimiter=',')
    logging.info('loaded all data {}'.format(data.shape))

    feature_info_name = 'autism.{}'.format(FEATURE_INFO_EXT)
    feature_info_path = os.path.join(args.datadir, feature_info_name)
    fnames, ftypes, fdomains = load_feature_info_preprocess(feature_info_path)
    logging.info('Loaded feature info file {}'.format(feature_info_path))

    cfdomains = []
    for i, (fn, ft, fd) in enumerate(zip(fnames, ftypes, fdomains)):

        if ft == 'categorical':
            if fd is None:
                fd = estimate_categorical_domain(data, i)

        elif ft == 'continuous':
            if fd is None:
                fd = estimate_continuous_domain(data, i,
                                                range=(data[:, i].min() - 0.01,
                                                       data[:, i].max() + 0.01),
                                                binning_method=bin_method)

        elif ft == 'discrete':
            if fd is None:
                fd = estimate_categorical_domain(data, i)

        #     try:
        #         _ = [float(k) for k, v in fd.items()]
        #     except:
        #         logging.info('Cannot convert discrete domain to float {}'.format(fd))
        #         fd = {str(v): v for _k, v in fd.items()}
        cfdomains.append(fd)

    #
    # feature info
    proc_feature_info_path = os.path.join(args.output, 'autism.{}'.format(FEATURE_INFO_EXT))
    save_feature_info_dict(fnames,
                           ftypes,
                           cfdomains,
                           proc_feature_info_path, domain_keys=False, range=False)

    #
    # splitting train and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_perc, random_state=rand_gen)

    target_feature = args.target
    target_data = data[:, target_feature]
    train = None
    valid = None
    test = None
    for train_index, test_index in sss.split(data, target_data):
        train = data[train_index]
        test = data[test_index]

    logging.info('Created train and test splits:\n\ttrain:\t{}\n\ttest:\t{}'.format(train.shape,
                                                                                    test.shape))

    if args.valid_perc > 0.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.valid_perc, random_state=rand_gen)
        for train_index, valid_index in sss.split(train, train[:, target_feature]):
            train_d = train[train_index]
            valid = train[valid_index]

        logging.info('Created train valid and test splits:\n\ttrain:\t{}\n\tvalid:\t{}\n\ttest:\t{}'.format(train_d.shape,
                                                                                                            valid.shape,
                                                                                                            test.shape))

    #
    # processing data
    train = uci_data_to_numpy(train_d,
                              ftypes,
                              cfdomains,
                              # sep=',',
                              # invis='\n.',
                              # header=None,
                              dtype=numpy.float64)

    #
    #
    # saving them
    feature_formats = get_feature_formats(ftypes)

    train_path = os.path.join(args.output, 'autism.{}.{}'.format(TRAIN_DATA_EXT, MLC_TRAIN_EXT))
    numpy.savetxt(train_path, train, fmt=feature_formats, delimiter=',')
    logging.info('Dumped training data to {}'.format(train_path))

    if valid is not None:
        valid = uci_data_to_numpy(valid,
                                  ftypes,
                                  cfdomains,
                                  # sep=',',
                                  # invis='\n.',
                                  # header=None,
                                  dtype=numpy.float64)
        valid_path = os.path.join(args.output,
                                  'autism.{}.{}'.format(VALID_DATA_EXT,
                                                        MLC_TRAIN_EXT))
        numpy.savetxt(valid_path, valid, fmt=feature_formats, delimiter=',')
        logging.info('Dumped validation data to {}'.format(valid_path))

    test = uci_data_to_numpy(test,
                             ftypes,
                             cfdomains,
                             # sep=',',
                             # invis='\n.',
                             # header=None,
                             dtype=numpy.float64)

    test_path = os.path.join(args.output, 'autism.{}.{}'.format(TEST_DATA_EXT, MLC_TRAIN_EXT))
    numpy.savetxt(test_path, test, fmt=feature_formats, delimiter=',')
    logging.info('Dumped test data to {}'.format(test_path))

    print(numpy.isnan(train_d).sum())
    print(numpy.isnan(valid).sum())
    print(numpy.isnan(test).sum())
