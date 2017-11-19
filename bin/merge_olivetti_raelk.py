import os
import logging
import gzip
import pickle
import argparse
from time import perf_counter
import operator

import numpy
from numpy.testing import assert_array_equal

import matplotlib

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

from keras.models import load_model


SPLIT_NAMES = ['train', 'valid', 'test']


def plot_digit(image_data, img_size=(64, 64), fig_size=(4, 4), output=None, show=True):

    matrix_data = image_data.reshape(img_size)

    fig, ax = pyplot.subplots(figsize=fig_size)
    ax.imshow(matrix_data, cmap=pyplot.get_cmap('gray'))

    if show:
        pyplot.show()

    if output:
        pp = PdfPages(output + '.pdf')
        pp.savefig(fig)
        pp.close()
        logging.info('Saved image to pdf {}'.format(output))


RED_CMAP = pyplot.get_cmap('Reds_r')
BLUE_CMAP = pyplot.get_cmap('Blues_r')


def plot_digits_matrix(images,
                       m, n,
                       img_size=(64, 64),
                       fig_size=(12, 12),
                       output=None, dpi=300,
                       w_space=0.0,
                       h_space=0.0,
                       cmap=pyplot.get_cmap('gray'),
                       masking=None,
                       mask_cmap=RED_CMAP,
                       show=True):

    import matplotlib.gridspec as gridspec

    gs1 = gridspec.GridSpec(m, n)
    gs1.update(wspace=w_space, hspace=h_space)
    # print(gs1)

    # print(len(images))
    fig = pyplot.figure(figsize=fig_size, dpi=dpi)
    for x in range(m):
        for y in range(n):
            id = n * x + y
            if id < len(images):
                # print(id, n, x, y)
                ax = fig.add_subplot(gs1[id])
                img_data = images[id]
                if masking is not None:
                    mask = numpy.zeros(img_data.shape, dtype=bool)
                    mask[masking] = True
                    # print(masking)
                    # print(mask)
                    img_1 = numpy.ma.masked_array(img_data, ~mask).reshape(img_size)
                    ax.imshow(img_1, cmap=mask_cmap)
                    img_2 = numpy.ma.masked_array(img_data, mask).reshape(img_size)
                    ax.imshow(img_2, cmap=cmap)
                else:
                    img = img_data.reshape(img_size)
                    ax.imshow(img, cmap=cmap)
                pyplot.xticks(numpy.array([]))
                pyplot.yticks(numpy.array([]))

    # pyplot.tight_layout()
    pyplot.subplots_adjust(left=None, right=None, wspace=w_space, hspace=h_space)
    if output:
        pp = PdfPages(output + '.pdf')
        pp.savefig(fig)
        pp.close()
        logging.info('Saved image to pdf {}'.format(output))

    if show:
        pyplot.show()


def load_cv_splits(dataset_path,
                   dataset_name,
                   n_folds,
                   train_ext=None, valid_ext=None, test_ext=None,
                   x_only=False,
                   y_only=False,
                   dtype='int32'):

    if x_only and y_only:
        raise ValueError('Both x and y only specified')

    logging.info('Expecting dataset into {} folds for {}'.format(n_folds, dataset_name))
    fold_splits = []

    if (train_ext is not None and test_ext is not None):
        #
        # NOTE: this applies only to x-only/y-only data files
        for i in range(n_folds):
            logging.info('Looking for train-test split {}'.format(i))

            train_path = '{}.{}.{}'.format(dataset_path, i, train_ext)
            logging.info('Loading training csv file {}'.format(train_path))
            train = numpy.loadtxt(train_path, dtype=dtype, delimiter=',')

            test_path = '{}.{}.{}'.format(dataset_path, i, test_ext)
            logging.info('Loading test csv file {}'.format(test_path))
            test = numpy.loadtxt(test_path, dtype=dtype, delimiter=',')

            assert train.shape[1] == test.shape[1]

            fold_splits.append((train, None, test))
    else:
        logging.info('Trying to load pickle file {}'.format(dataset_path))
        #
        # trying to load a pickle file containint k = n_splits
        # [((train_x,  train_y), (test_x, test_y))_1, ... ((train_x, train_y), (test_x, test_y))_k]

        fsplit = None
        if dataset_path.endswith('.pklz'):
            fsplit = gzip.open(dataset_path, 'rb')
        else:
            fsplit = open(dataset_path, 'rb')

        folds = pickle.load(fsplit)
        fsplit.close()

        assert len(folds) == n_folds

        for splits in folds:

            if len(splits) == 1:
                raise ValueError('Not expecting a fold made by a single split')
            elif len(splits) == 2:
                train_split, test_split = splits
                #
                # do they contain label information?
                if x_only and len(train_split) == 2 and len(test_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    fold_splits.append((train_x, None, test_x))
                elif y_only and len(train_split) == 2 and len(test_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    fold_splits.append((train_y, None, test_y))
                else:
                    fold_splits.append((train_split, None, test_split))
            elif len(splits) == 3:
                train_split, valid_split, test_split = splits
                if x_only and len(train_split) == 2 and len(test_split) == 2 and len(valid_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    valid_x, valid_y = valid_split
                    fold_splits.append((train_x, valid_x, test_x))
                elif y_only and len(train_split) == 2 and len(test_split) == 2 and len(valid_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    valid_x, valid_y = valid_split
                    fold_splits.append((train_y, valid_y, test_y))
                else:
                    fold_splits.append((train_split, valid_split, test_split))

    assert len(fold_splits) == n_folds
    # logging.info('Loaded folds for {}'.format(dataset_name))
    # for i, (train, valid, test) in enumerate(fold_splits):
    #     logging.info('\tfold:\t{} {} {} {} '.format(i, len(train), len(test), valid))
    #     if len(train) == 2 and len(test) == 2:
    #         logging.info('\t\ttrain x:\tsize: {}\ttrain y:\tsize: {}'.format(train[0].shape,
    #                                                                          train[1].shape))
    #         logging.info('\t\ttest:\tsize: {}\ttest:\tsize: {}'.format(test[0].shape,
    #                                                                    test[1].shape))
    #     else:
    #         logging.info('\t\ttrain:\tsize: {}'.format(train.shape))
    #         logging.info('\t\ttest:\tsize: {}'.format(test.shape))

    return fold_splits


def load_train_val_test_splits(dataset_path,
                               dataset_name,
                               train_ext=None, valid_ext=None, test_ext=None,
                               x_only=False,
                               y_only=False,
                               dtype='int32'):

    if x_only and y_only:
        raise ValueError('Both x and y only specified')

    logging.info('Looking for (train/valid/test) dataset splits: %s', dataset_path)

    if train_ext is not None:
        #
        # NOTE this works only with x-only data files
        train_path = '{}.{}'.format(dataset_path, train_ext)
        logging.info('Loading training csv file {}'.format(train_path))
        train = numpy.loadtxt(train_path, dtype='int32', delimiter=',')

        if valid_ext is not None:
            valid_path = '{}.{}'.format(dataset_path, valid_ext)
            logging.info('Loading valid csv file {}'.format(valid_path))
            valid = numpy.loadtxt(valid_path, dtype='int32', delimiter=',')
            assert train.shape[1] == valid.shape[1]

        if test_ext is not None:
            test_path = '{}.{}'.format(dataset_path, test_ext)
            logging.info('Loading test csv file {}'.format(test_path))
            test = numpy.loadtxt(test_path, dtype='int32', delimiter=',')
            assert train.shape[1] == test.shape[1]

    else:
        logging.info('Trying to load pickle file {}'.format(dataset_path))
        #
        # trying to load a pickle containing (train_x) | (train_x, test_x) |
        # (train_x, valid_x, test_x)
        fsplit = None
        if dataset_path.endswith('.pklz'):
            fsplit = gzip.open(dataset_path, 'rb')
        else:
            fsplit = open(dataset_path, 'rb')

        splits = pickle.load(fsplit)
        fsplit.close()

        if len(splits) == 1:
            logging.info('Only training set')
            train = splits
            if x_only and isinstance(train, tuple):
                logging.info('\tonly x')
                train = train[0]

        elif len(splits) == 2:
            logging.info('Found training and test set')
            train, test = splits

            if len(train) == 2 and len(test) == 2:
                assert train[0].shape[1] == test[0].shape[1]
                assert train[1].shape[1] == test[1].shape[1]
                assert train[0].shape[0] == train[1].shape[0]
                assert test[0].shape[0] == test[1].shape[0]
            else:
                assert train.shape[1] == test.shape[1]

            if x_only:
                logging.info('\tonly x')
                if isinstance(train, tuple) and isinstance(test, tuple):
                    train = train[0]
                    test = test[0]
                else:
                    raise ValueError('Cannot get x only for train and test splits')
            elif y_only:
                logging.info('\tonly y')
                if isinstance(train, tuple) and isinstance(test, tuple):
                    train = train[1]
                    test = test[1]
                else:
                    raise ValueError('Cannot get y only for train and test splits')

        elif len(splits) == 3:
            logging.info('Found training, validation and test set')
            train, valid, test = splits

            if len(train) == 2 and len(test) == 2 and len(valid) == 2:
                assert train[0].shape[1] == test[0].shape[1]
                assert train[0].shape[1] == valid[0].shape[1]
                if train[1].ndim > 1 and test[1].ndim > 1 and valid[1].ndim > 1:
                    assert train[1].shape[1] == test[1].shape[1]
                    assert train[1].shape[1] == valid[1].shape[1]
                assert train[0].shape[0] == train[1].shape[0]
                assert test[0].shape[0] == test[1].shape[0]
                assert valid[0].shape[0] == valid[1].shape[0]

                if x_only:
                    logging.info('\tonly x')
                    if isinstance(train, tuple) and \
                       isinstance(test, tuple) and \
                       isinstance(valid, tuple):

                        train = train[0]
                        valid = valid[0]
                        test = test[0]
                    else:
                        raise ValueError('Cannot get x only for train, valid and test splits')
                elif y_only:
                    logging.info('\tonly y')
                    if isinstance(train, tuple) and \
                       isinstance(test, tuple) and \
                       isinstance(valid, tuple):

                        train = train[1]
                        valid = valid[1]
                        test = test[1]
                    else:
                        raise ValueError('Cannot get y only for train, valid and test splits')
            else:
                assert train.shape[1] == test.shape[1]
                assert train.shape[1] == valid.shape[1]

        else:
            raise ValueError('More than 3 splits, check pkl file {}'.format(dataset_path))

    fold_splits = [(train, valid, test)]

    logging.info('Loaded dataset {}'.format(dataset_name))

    return fold_splits


def print_fold_splits_shapes(fold_splits):
    for f, fold in enumerate(fold_splits):
        logging.info('\tfold {}'.format(f))
        for s, split in enumerate(fold):
            if split is not None:
                split_name = SPLIT_NAMES[s]
                if len(split) == 2:
                    split_x, split_y = split
                    logging.info('\t\t{}\tx: {}\ty: {}'.format(split_name,
                                                               split_x.shape, split_y.shape))
                else:
                    logging.info('\t\t{}\t(x/y): {}'.format(split_name,
                                                            split.shape))


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


def compute_histogram(data,
                      bins='unique',
                      range=None,
                      density=False):
    """
    Just a wrapper around numpy.histogram
    """
    #
    # using numpy
    # else:
    return numpy.histogram(data, bins=bins, range=range, density=density)


def estimate_continuous_domain(tdata,
                               feature_id,
                               binning_method='unique',
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--h1', type=str,
                        help='Path of first half of data')

    parser.add_argument('--h2', type=str,
                        help='Path of second half of data')
    # parser.add_argument("dataset", type=str,
    #                     help='(MLC) dataset name')

    parser.add_argument('--data-exts', type=str, nargs='+',
                        default=None,
                        help='Dataset split extensions')

    parser.add_argument('--dtype', type=str, nargs='?',
                        default='int32',
                        help='Loaded dataset type')

    parser.add_argument('--bins', type=str, nargs='?',
                        default='unique',
                        help='Binning type')

    parser.add_argument("--data-file", type=str, nargs='?',
                        default='raelk.mnist.pklz',
                        help='Specify dataset dir')

    parser.add_argument("-o", "--output", type=str, nargs='?',
                        default='./mlutils/datasets/olivetti/',
                        help='output path')

    parser.add_argument("--masks", type=str, nargs='+',
                        # default=['up', 'down', 'left', 'right'],
                        default=[],
                        help='Specify which masked data to create')

    parser.add_argument('--cv', type=int,
                        help='Folds for cross validation for model selection')

    parser.add_argument('--y-only', action='store_true',
                        help='Whether to load only the Y from the model pickle file')

    parser.add_argument("--seed", type=int, nargs='?',
                        default=1337,
                        help='Random seed')

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

    logger.info("Starting with arguments:\n%s", args)

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    dataset_name_h1 = args.h1.split('/')[-1]
    #
    # replacing  suffixes names
    dataset_name_h1 = dataset_name_h1.replace('.pklz', '')
    dataset_name_h1 = dataset_name_h1.replace('.pkl', '')
    dataset_name_h1 = dataset_name_h1.replace('.pickle', '')

    dataset_name_h2 = args.h1.split('/')[-1]
    #
    # replacing  suffixes names
    dataset_name_h2 = dataset_name_h2.replace('.pklz', '')
    dataset_name_h2 = dataset_name_h2.replace('.pkl', '')
    dataset_name_h2 = dataset_name_h2.replace('.pickle', '')

    train_ext = None
    valid_ext = None
    test_ext = None
    repr_train_ext = None
    repr_valid_ext = None
    repr_test_ext = None

    if args.data_exts is not None:
        if len(args.data_exts) == 1:
            train_ext, = args.data_exts
        elif len(args.data_exts) == 2:
            train_ext, test_ext = args.data_exts
        elif len(args.data_exts) == 3:
            train_ext, valid_ext, test_ext = args.data_exts
        else:
            raise ValueError('Up to 3 data extenstions can be specified')

    n_folds = args.cv if args.cv is not None else 1

    x_only = None
    y_only = None
    if args.y_only:
        x_only = False
        y_only = True
    else:
        x_only = True
        y_only = False

    #
    # loading data and learned representations
    fold_splits_h1 = None
    if args.cv is not None:
        fold_splits_h1 = load_cv_splits(args.h1,
                                        dataset_name_h1,
                                        n_folds,
                                        train_ext=train_ext,
                                        valid_ext=valid_ext,
                                        test_ext=test_ext,
                                        x_only=x_only,
                                        y_only=y_only,
                                        dtype=args.dtype)

    else:
        fold_splits_h1 = load_train_val_test_splits(args.h1,
                                                    dataset_name_h1,
                                                    train_ext=train_ext,
                                                    valid_ext=valid_ext,
                                                    test_ext=test_ext,
                                                    x_only=x_only,
                                                    y_only=y_only,
                                                    dtype=args.dtype)

    #
    # printing
    print_fold_splits_shapes(fold_splits_h1)

    #
    # loading data and learned representations
    fold_splits_h2 = None
    if args.cv is not None:
        fold_splits_h2 = load_cv_splits(args.h2,
                                        dataset_name_h2,
                                        n_folds,
                                        train_ext=train_ext,
                                        valid_ext=valid_ext,
                                        test_ext=test_ext,
                                        x_only=x_only,
                                        y_only=y_only,
                                        dtype=args.dtype)

    else:
        fold_splits_h2 = load_train_val_test_splits(args.h2,
                                                    dataset_name_h2,
                                                    train_ext=train_ext,
                                                    valid_ext=valid_ext,
                                                    test_ext=test_ext,
                                                    x_only=x_only,
                                                    y_only=y_only,
                                                    dtype=args.dtype)

    #
    # printing
    print_fold_splits_shapes(fold_splits_h2)

    train_h1, valid_h1, test_h1 = fold_splits_h1[0]
    train_h2, valid_h2, test_h2 = fold_splits_h2[0]

    train_x = numpy.concatenate([train_h1, train_h2], axis=1)
    valid_x = numpy.concatenate([valid_h1, valid_h2], axis=1)
    test_x = numpy.concatenate([test_h1, test_h2], axis=1)

    print('Concatenated mask shapes train:{} valid:{} test:{}'.format(train_x.shape,
                                                                      valid_x.shape,
                                                                      test_x.shape))

    n_features = train_x.shape[1]
    assert valid_x.shape[1] == n_features
    assert test_x.shape[1] == n_features

    feature_names = ['ae_{}'.format(i) for i in range(n_features)]

    feature_types = ['continuous' for i in range(n_features)]
    print('feature types', feature_types)

    #
    # numpy.array
    full_x = numpy.concatenate([train_x, valid_x, test_x], axis=0)
    print('fully dataset sizes', full_x.shape)

    print('Estimating domains by embeddings')
    # domains_x = estimate_domains_range(full_x, feature_types_x)
    bin_method = args.bins
    bin_range_width = 0.001
    domains = [estimate_continuous_domain(full_x, i,
                                          range=(full_x[:, i].min() - bin_range_width,
                                                 full_x[:, i].max() + bin_range_width),
                                          binning_method=bin_method) for i in range(n_features)]
    print('domains', domains)

    out_feature_info_path = os.path.join(out_path, 'aug.raelk.features')
    save_feature_info_dict(feature_names, feature_types,
                           domains, out_feature_info_path, range=False)
    print('Saved feature info file to ', out_feature_info_path)

    #
    # saving modifyied dataset in text format

    out_train_path = os.path.join(out_path, 'aug.raelk.train.data')
    out_valid_path = os.path.join(out_path, 'aug.raelk.valid.data')
    out_test_path = os.path.join(out_path, 'aug.raelk.test.data')

    feature_formats = get_feature_formats(feature_types)
    numpy.savetxt(out_train_path, train_x, fmt=feature_formats, delimiter=',')
    numpy.savetxt(out_valid_path, valid_x, fmt=feature_formats, delimiter=',')
    numpy.savetxt(out_test_path, test_x, fmt=feature_formats, delimiter=',')
    print('augmented splits dumped to:\n\t{}\n\t{}\n\t{}'.format(out_train_path,
                                                                 out_valid_path,
                                                                 out_test_path))
