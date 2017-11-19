import sys
import os
import logging
from time import perf_counter
import argparse

import numpy
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

DATA_EXT = 'data'
TRAIN_DATA_EXT = 'train.{}'.format(DATA_EXT)
VALID_DATA_EXT = 'valid.{}'.format(DATA_EXT)
TEST_DATA_EXT = 'test.{}'.format(DATA_EXT)

ALL_MLC_DATASETS = ['anneal',
                    'anneal-U',
                    'australian',
                    'auto',
                    'balance-scale',
                    'breast',
                    'breast-cancer',
                    'cars',
                    'chess',
                    'cleve',
                    'crx',
                    'diabetes',
                    'german',
                    'german-org',
                    'glass',
                    'glass2',
                    'heart',
                    'hepatitis',
                    'horse-colic',
                    'hypothyroid',
                    'ionosphere',
                    'iris',
                    'labor-neg',
                    'lenses',
                    'letter',
                    'monk1',
                    'mushroom',
                    'solar'
                    'sonar',
                    'soybean',
                    'vote',
                    'wine',
                    'zoo']


MLC_DATASETS = ['anneal-U',
                'australian',
                'auto',
                'balance-scale',
                'breast',
                'breast-cancer',
                'cars',
                # 'chess',
                'cleve',
                'crx',
                'diabetes',
                'german',
                'german-org',
                'glass',
                'glass2',
                'heart',
                'hepatitis',
                'iris',
                'solar'
                ]

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

    parser.add_argument('-d', "--datasets", type=str, nargs='+',
                        default=MLC_DATASETS,
                        help='Dataset file names')

    parser.add_argument('--data-exts', type=str, nargs='+',
                        default=[MLC_TRAIN_EXT, MLC_TEST_EXT],
                        help='Dataset split extensions')

    parser.add_argument('-o', '--output', type=str, nargs='?',
                        default='./data/',
                        help='Output dir path')

    parser.add_argument('-b', '--bins', type=str, nargs='?',
                        default='blocks',
                        help='Binning method (blocks|unique|auto|fd|...)')

    parser.add_argument('--valid-perc', type=float,
                        default=0.1,
                        help='Percentage of training set to reserve for validation')

    parser.add_argument('--miss-val', type=str,
                        default=MISS_VAL_STR,
                        help='Missing value string representation')

    parser.add_argument('--header', type=int,
                        default=HEADER,
                        help='Header length')

    parser.add_argument('-v', '--verbose', type=int, nargs='?',
                        default=1,
                        help='Verbosity level')

    # parsing the args
    args = parser.parse_args()

    #
    # setting verbosity level
    if args.verbose == 1:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    elif args.verbose == 2:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logging.info("Starting with arguments:\n%s", args)

    #
    # for all datasets, remove samples containing missing values, generate new versions
    # and store them
    miss_proc_dir = os.path.join(args.output, NON_MISS_STR)
    os.makedirs(miss_proc_dir, exist_ok=True)
    for d in args.datasets:
        logging.info('\n\nProcessing dataset {}'.format(d))

        #
        # loading feature info files
        feature_info_name = '{}.{}'.format(d, FEATURE_INFO_EXT)
        feature_info_path = os.path.join(args.datadir, feature_info_name)
        fnames, ftypes, fdomains = load_feature_info_preprocess(feature_info_path)
        logging.info('Loaded feature info file {}'.format(feature_info_path))

        tdata = []
        for ext in args.data_exts:
            split_name = '{}.{}'.format(d, ext)
            data_path = os.path.join(args.datadir, split_name)
            logging.info('\tlooking for file split {}'.format(data_path))

            non_miss_split_name = '{}.{}.{}'.format(d, NON_MISS_STR, ext)
            non_miss_path = os.path.join(miss_proc_dir, non_miss_split_name)
            remove_missing_value_samples(data_path,
                                         missing_value_str=args.miss_val,
                                         header=args.header,
                                         output=non_miss_path)

            tdata.extend(load_uci_data_from_text(non_miss_path,
                                                 sep=',',
                                                 invis='\n.',
                                                 header=None))

        cfdomains = []
        for i, (fn, ft, fd) in enumerate(zip(fnames, ftypes, fdomains)):

            if ft == 'categorical':
                if fd is None:
                    fd = estimate_categorical_domain(tdata, i)

            elif ft == 'discrete':
                if fd is None:
                    fd = estimate_categorical_domain(tdata, i)

            elif ft == 'continuous':
                if fd is None:
                    fd = estimate_continuous_domain_range(tdata, i)

            # elif ft == 'discrete':
            #     try:
            #         _ = [float(k) for k, v in fd.items()]
            #     except:
            #         logging.info('Cannot convert discrete domain to float {}'.format(fd))
            #         fd = {str(v): v for _k, v in fd.items()}
            cfdomains.append(fd)

        #
        # writing back the domain for the missing values
        feature_info_name = '{}.{}.{}'.format(d, NON_MISS_STR, FEATURE_INFO_EXT)
        miss_feature_info_path = os.path.join(miss_proc_dir, feature_info_name)
        save_feature_info_dict(fnames, ftypes, cfdomains, miss_feature_info_path, domain_keys=True)

        #
        # splitting for a validation set
        splits, split_exts = [], []
        train_split_name = '{}.{}.{}'.format(d, NON_MISS_STR, MLC_TRAIN_EXT)
        train_path = os.path.join(miss_proc_dir, train_split_name)
        train = load_uci_data_from_text(train_path,
                                        sep=',',
                                        invis='\n.',
                                        header=None)
        if args.valid_perc > 0:

            n_train_samples = len(train)
            n_valid_samples = int(args.valid_perc * n_train_samples)
            assert n_valid_samples > 0
            logging.info('Reserving {} samples for validation'.format(n_valid_samples))

            valid = train[:n_valid_samples]
            train = train[n_valid_samples:]

            splits.append(train)
            split_exts.append(TRAIN_DATA_EXT)

            splits.append(valid)
            split_exts.append(VALID_DATA_EXT)

        else:
            splits.append(train)
            split_exts.append(TRAIN_DATA_EXT)

        test_split_name = '{}.{}.{}'.format(d, NON_MISS_STR, MLC_TEST_EXT)
        test_path = os.path.join(miss_proc_dir, test_split_name)
        test = load_uci_data_from_text(test_path,
                                       sep=',',
                                       invis='\n.',
                                       header=None)
        splits.append(test)
        split_exts.append(TEST_DATA_EXT)

        proc_dir = os.path.join(args.output, PROC_STR, args.bins)
        os.makedirs(proc_dir, exist_ok=True)
        for s, ext in zip(splits, split_exts):
            #
            # convert data to numpy array
            proc_data = uci_data_to_numpy(s,
                                          ftypes,
                                          cfdomains,
                                          # sep=',',
                                          # invis='\n.',
                                          # header=None,
                                          dtype=numpy.float64)

            #
            # now serialize it
            proc_split_name = '{}.{}'.format(d, ext)
            proc_path = os.path.join(proc_dir, proc_split_name)

            feature_formats = get_feature_formats(ftypes)

            numpy.savetxt(proc_path, proc_data, fmt=feature_formats, delimiter=',')

        pfdomains = []
        for i, (fn, ft, fd) in enumerate(zip(fnames, ftypes, cfdomains)):

            if ft == 'continuous':
                r = (fd['min'], fd['max'])
                print('range', r)
                #
                # using astropy
                fd = estimate_continuous_domain(train, i, range=r, binning_method=args.bins)
                # fd = estimate_continuous_domain_min_max(train, i, min=fd['min'], max=fd['max'])
                print(fd)
            pfdomains.append(fd)

        #
        # writing back the domain for the missing values
        feature_info_name = '{}.{}'.format(d, FEATURE_INFO_EXT)
        proc_feature_info_path = os.path.join(proc_dir, feature_info_name)
        save_feature_info_dict(fnames,
                               ftypes,
                               pfdomains,
                               proc_feature_info_path, domain_keys=False, range=False)

    logging.info('\n\nProcessed all {} datasets'.format(len(args.datasets)))
