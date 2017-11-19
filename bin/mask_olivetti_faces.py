import os
import logging
import gzip
import pickle
import argparse
from time import perf_counter

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


def save_splits_raelk_format(x_train,
                             x_valid,
                             x_test,
                             output=None,
                             file_name='raelk.olivetti.pklz'):
    fold = [x_train, x_valid, x_test]
    print(x_train[:1])

    # repr_save_path = os.path.join(out_dir, out_path)

    os.makedirs(output, exist_ok=True)
    repr_save_path = os.path.join(output, file_name)
    with gzip.open(repr_save_path, 'wb') as f:
        print('Saving splits to {}'.format(repr_save_path))
        pickle.dump(fold, f, protocol=4)
        print('Saved.')


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


def filter_image_data(data, n_cols=64, mask=None):

    feature_mask = None
    n_features = data.shape[1]

    # if mask == 'left':
    #     feature_mask = left_mask(n_features, n_cols)
    # elif mask == 'right':
    #     feature_mask = right_mask(n_features, n_cols)
    # elif mask == 'up':
    #     feature_mask = upper_mask(n_features)
    # elif mask == 'down':
    #     feature_mask = lower_mask(n_features)

    feature_mask = get_image_mask(n_features, n_cols=n_cols, mask=mask)

    filtered_data = data[:, feature_mask]
    logging.info('--> Filtered data --> {}'.format(filtered_data.shape))

    return filtered_data


def decode_predictions(repr_preds, ae_decoder):

    preds = ae_decoder.predict(repr_preds)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help='(MLC) dataset name')

    parser.add_argument('--data-exts', type=str, nargs='+',
                        default=None,
                        help='Dataset split extensions')

    parser.add_argument('--dtype', type=str, nargs='?',
                        default='int32',
                        help='Loaded dataset type')

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

    dataset_name = args.dataset.split('/')[-1]
    #
    # replacing  suffixes names
    dataset_name = dataset_name.replace('.pklz', '')
    dataset_name = dataset_name.replace('.pkl', '')
    dataset_name = dataset_name.replace('.pickle', '')

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
    if args.cv is not None:
        fold_splits = load_cv_splits(args.dataset,
                                     dataset_name,
                                     n_folds,
                                     train_ext=train_ext,
                                     valid_ext=valid_ext,
                                     test_ext=test_ext,
                                     x_only=x_only,
                                     y_only=y_only,
                                     dtype=args.dtype)

    else:
        fold_splits = load_train_val_test_splits(args.dataset,
                                                 dataset_name,
                                                 train_ext=train_ext,
                                                 valid_ext=valid_ext,
                                                 test_ext=test_ext,
                                                 x_only=x_only,
                                                 y_only=y_only,
                                                 dtype=args.dtype)

    #
    # printing
    print_fold_splits_shapes(fold_splits)

    # n_instances = train.shape[0]
    # n_test_instances = test.shape[0]

    assert len(fold_splits) == 1
    orig_train_x, orig_valid_x, orig_test_x, = fold_splits[0]

    n_cols = 64

    masked_img_sizes = {'left': (64, 32), 'right': (64, 32), 'up': (32, 64), 'down': (32, 64)}

    for i, m in enumerate(args.masks):

        print('\n\nProcessing mask {}\n\n'.format(m))

        masked_train_split_x = filter_image_data(orig_train_x, n_cols=n_cols, mask=m)
        # orig_train_x_splits.append(masked_train_split_x)
        print('masked train shape', masked_train_split_x.shape)

        masked_valid_split_x = filter_image_data(orig_valid_x, n_cols=n_cols, mask=m)
        # orig_valid_x_splits.append(masked_valid_split_x)
        print('masked valid shape', masked_valid_split_x.shape)

        masked_test_split_x = filter_image_data(orig_test_x, n_cols=n_cols, mask=m)
        # orig_test_x_splits.append(masked_test_split_x)
        print('masked test shape', masked_test_split_x.shape)

        #
        # saving them
        masked_out_path = os.path.join(out_path,  m)
        save_splits_raelk_format(masked_train_split_x,
                                 masked_valid_split_x,
                                 masked_test_split_x,
                                 output=masked_out_path)
        print('Saved masked fold set to {}'.format(masked_out_path))

        plot_digit(masked_train_split_x[0], img_size=masked_img_sizes[m],
                   fig_size=(4, 4), output=None, show=True)
