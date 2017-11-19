import os
import logging
import gzip
import pickle
import argparse
from time import perf_counter

import numpy
from numpy.testing import assert_array_equal


from keras.datasets import mnist
from keras.models import load_model

from tfspn.piecewise import estimate_domains, estimate_domains_range
from mlutils.datasets import load_feature_info_preprocess
from mlutils.datasets import save_feature_info
from mlutils.datasets import save_feature_info_dict
from mlutils.datasets import get_feature_formats
from mlutils.datasets import estimate_continuous_domain
from mlutils.datasets import estimate_categorical_domain

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

SPLIT_NAMES = ['train', 'valid', 'test']
SCORE_NAMES = {'accuracy': 'acc',
               'hamming': 'ham',
               'exact': 'exc',
               'jaccard': 'jac',
               'micro-f1': 'mif',
               'macro-f1': 'maf',
               'micro-auc-pr': 'mipr',
               'macro-auc-pr': 'mapr', }


def plot_digit(image_data, img_size=(28, 28), fig_size=(4, 4), output=None, show=True):

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
                       img_size=(28, 28),
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

    # matrix_data = image_data.reshape(img_size)


def plot_embedding(X, y, X_embedded, name, min_dist=10.0,
                   marker='x',
                   marker_size=40,
                   marker_width=2,
                   fig_size=(16, 16),
                   img_size=(28, 28),
                   max_instances=5000,
                   zoom=1.0,
                   rand_gen=None,
                   output=None,
                   invert=None,
                   show=True):

    class_cmap = None
    if y is not None:
        assert X.shape[0] == y.shape[0], "Different number of instances X:{} y:{}".format(X.shape,
                                                                                          y.shape)
        class_cmap = pyplot.get_cmap('tab20')
        y_norm = (y - y.min()) / (y.max() - y.min())

    img_cmap = None
    if invert:
        img_cmap = pyplot.get_cmap('binary')
    else:
        img_cmap = pyplot.get_cmap('binary_r')

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(1337)

    fig = pyplot.figure(figsize=fig_size)
    ax = pyplot.axes(frameon=False)
    # pyplot.title()
    pyplot.setp(ax, xticks=(), yticks=())
    pyplot.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                           wspace=0.0, hspace=0.0)
    pyplot.scatter(X_embedded[:, 0], X_embedded[:, 1],
                   c=y,
                   cmap=matplotlib.cm.Spectral,
                   marker=marker,
                   s=marker_size,
                   linewidth=marker_width)

    if min_dist is not None:
        from matplotlib import offsetbox
        shown_images = numpy.array([[15., 15.]])
        indices = numpy.arange(X_embedded.shape[0])
        rand_gen.shuffle(indices)
        for i in indices[:max_instances]:
            dist = numpy.sum((X_embedded[i] - shown_images) ** 2, 1)
            if numpy.min(dist) < min_dist:
                continue

            class_color_dict = {}
            if y is not None:
                class_color_dict = dict(fc=class_cmap(y_norm[i]),
                                        width=5.0)

            shown_images = numpy.r_[shown_images, [X_embedded[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(*img_size),
                                      cmap=img_cmap,
                                      zoom=zoom),
                X_embedded[i],
                pad=0.5,
                bboxprops=class_color_dict)
            ax.add_artist(imagebox)

    if show:
        pyplot.show()

    if output is not None:
        pp = PdfPages(output + '.pdf')
        pp.savefig(fig, bbox_inches='tight', pad_inches=0)
        pp.close()
        print('pdf image saved to {}'.format(output))


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


def load_mnist_train_test_splits():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('Loaded train {} and test {} splits'.format(x_train.shape,
                                                      x_test.shape))
    print('\twith labels {} and {}'.format(y_train.shape, y_test.shape))

    #
    # reshaping and preprocessing
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), numpy.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), numpy.prod(x_test.shape[1:])))

    print('Processed train {} and test {} splits'.format(x_train.shape,
                                                         x_test.shape))

    #
    # saving a portion for validation
    n_valid_samples = 10000

    x_valid = x_train[:n_valid_samples]
    y_valid = y_train[:n_valid_samples]
    x_train = x_train[n_valid_samples:]
    y_train = y_train[n_valid_samples:]

    print('New train {} and valid {} splits'.format(x_train.shape,
                                                    x_valid.shape))
    print('\twith labels {} and {}'.format(y_train.shape, y_valid.shape))

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def save_mnist_split_raelk_format(x_train, y_train,
                                  x_valid, y_valid,
                                  x_test, y_test,
                                  # out_dir='./mlutils/datasets/',
                                  # out_path='mnist'
                                  output='./mlutils/datasets/mnist',
                                  file_name='raelk.mnist.pklz'):
    fold = [(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]
    print(x_train[:1], y_train[:1])

    # repr_save_path = os.path.join(out_dir, out_path)

    os.makedirs(output, exist_ok=True)
    repr_save_path = os.path.join(output, file_name)
    with gzip.open(repr_save_path, 'wb') as f:
        print('Saving splits to {}'.format(repr_save_path))
        pickle.dump(fold, f, protocol=4)
        print('Saved.')


def filter_samples_by_class(split_x, split_y,
                            classes):

    n_samples = split_x.shape[0]
    assert len(split_y) == n_samples

    for c in classes:

        sample_ids = split_y == c
        n_to_remove = sample_ids.sum()

        split_x = split_x[~sample_ids]
        split_y = split_y[~sample_ids]

        assert len(split_x) == (n_samples - n_to_remove)
        assert len(split_x) == len(split_y)

        logging.info('\n>>> Removed {} samples of class {}'.format(n_to_remove, c))

    return split_x, split_y


def load_train_val_test_raelk_splits(repr_path,
                                     data_path):
    """
    Loading representations learned from an autoencoder (only on X!)
    Then optionally attach label information, if available
    """
    print(repr_path)
    print(data_path)
    repr_fold_splits = load_train_val_test_splits(repr_path,
                                                  'mnist',
                                                  train_ext=None,
                                                  valid_ext=None,
                                                  test_ext=None,
                                                  x_only=False,
                                                  y_only=False,
                                                  dtype=numpy.float64)

    repr_train_x, repr_valid_x, repr_test_x = repr_fold_splits[0]
    print('Loaded train {} valid {} test {} autoencoder repr'.format(repr_train_x.shape,
                                                                     repr_valid_x.shape,
                                                                     repr_test_x.shape))
    fold_splits = load_train_val_test_splits(data_path,
                                             'mnist',
                                             train_ext=None,
                                             valid_ext=None,
                                             test_ext=None,
                                             x_only=False,
                                             y_only=False,
                                             dtype=numpy.float64)

    train, valid, test = fold_splits[0]
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test

    assert train_y.shape[0] == repr_train_x.shape[0]
    assert valid_y.shape[0] == repr_valid_x.shape[0]
    assert test_y.shape[0] == repr_test_x.shape[0]

    return (repr_train_x, train_y), (repr_valid_x, valid_y), (repr_test_x, test_y)


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


def get_image_mask(n_features, n_cols=28, mask=None):
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


def filter_image_data(data, n_cols=28, mask=None):

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


def has_vertical_stroke(num):
    if num in {1, 4, 7}:
        return 1
    else:
        return 0


def has_circle_stroke(num):
    if num in {0, 6, 8, 9}:
        return 1
    else:
        return 0


def has_curve_left(num):
    if num in {2, 3, 5, 8, 9}:
        return 1
    else:
        return 0


def has_curve_right(num):
    if num in {5, 6}:
        return 1
    else:
        return 0


def has_horizontal_stroke(num):
    if num in {2, 3, 4, 5, 7}:
        return 1
    else:
        return 0


def has_double_curve(num):
    if num in {3, 8}:
        return 1
    else:
        return 0


def is_four_or_nine(num):
    if num in {4, 9}:
        return 1
    else:
        return 0


ENC_RULE_DICT = {'vert-stroke': has_vertical_stroke,
                 'circle': has_circle_stroke,
                 'curve-left': has_curve_left,
                 'curve-right': has_curve_right,
                 'hori-stroke': has_horizontal_stroke,
                 'double-curve': has_double_curve,
                 'for-nine': is_four_or_nine}


def augment_labels(labels,
                   # label_rules=['vert-stroke',
                   #              'circle',
                   #              'curve-left',
                   #              'curve-right',
                   #              'hori-stroke'],
                   label_rule_funcs,
                   keep_original=False,
                   one_hot_encoding_classes=False,
                   int_dtype=numpy.int32):
    """
    From an array containing the class labels (of size (n_instances, ))
    derive a (multi) label matrix of size (n_instances, n_labels) by
    applying n_labels rules
    """

    n_instances = len(labels)
    # n_augmentations = len(label_rules)
    n_augmentations = len(label_rule_funcs)
    # assert n_augmentations > 1, "Provide more than one rule"

    n_classes = 10
    if keep_original:
        if one_hot_encoding_classes:
            #
            # NOTE: this is hard coded
            n_augmentations += n_classes
        else:
            n_augmentations += 1

    multi_labels = numpy.zeros((n_instances, n_augmentations), dtype=int_dtype)
    starting_feature = 0

    if keep_original:
        #
        # storing original class labels in first columns (one hot encoded)
        if one_hot_encoding_classes:
            classes = numpy.arange(n_classes, dtype=int_dtype)
            multi_labels[labels[:, None] == classes[None, :]] = 1
            assert multi_labels[:, n_classes:].sum() == 0
            starting_feature = n_classes
        else:
            multi_labels[:, 0] = labels
            starting_feature = 1

    # for j, rule in enumerate(label_rules):
    #     rule_func = ENC_RULE_DICT[rule]
    for j, rule_func in enumerate(label_rule_funcs):
        for i in range(n_instances):
            multi_labels[i, j + starting_feature] = rule_func(labels[i])

    return multi_labels


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


def rand_label_rule(n_classes, min_num, max_num, rand_gen):

    n_random_digits = rand_gen.choice(numpy.arange(min_num, max_num))
    random_digits = rand_gen.choice(numpy.arange(n_classes), size=n_random_digits, replace=False)

    return tuple(r for r in sorted(random_digits))


def get_rule_func(pattern):

    pattern = set(pattern)
    print(pattern)

    def rule(num):
        if num in pattern:
            return 1
        else:
            return 0

    return rule


def generate_random_label_rules(n_classes, n_rules, min_num_digits, max_num_digits, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(1337)

    generated_patterns = set()
    while len(generated_patterns) < n_rules:

        pattern = rand_label_rule(n_classes, min_num=min_num_digits,
                                  max_num=max_num_digits,
                                  rand_gen=rand_gen)

        #
        # already seen pattern, skip it!
        if pattern not in generated_patterns:

            logging.info('Randomly selected features {}'.format(pattern))
            generated_patterns.add(pattern)

            # def pattern_rule(num):
            #     if num in set(pattern):
            #         return 1
            #     else:
            #         return 0

    rule_funcs = [get_rule_func(p) for p in generated_patterns]
    return rule_funcs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("dataset", type=str,
    #                     help='(MLC) dataset name')

    parser.add_argument("--data-file", type=str, nargs='?',
                        default='raelk.mnist.pklz',
                        help='Specify dataset dir')

    parser.add_argument("--ae-dir", type=str, nargs='?',
                        default='/home/valerio/Petto Redigi/trash/raelk.mnist_2017-05-22_15-00-18/',
                        help='Specify autoencoder dir')

    parser.add_argument("--repr-file", type=str, nargs='?',
                        default='ae.raelk.mnist.repr-data.pklz',
                        help='Specify autoencoder repr filename')

    parser.add_argument("--stub-feature-info-file", type=str, nargs='?',
                        default=None,
                        help='Specify the path to a feature file that has already estimated domains for embeddings')

    parser.add_argument('--keep-class', action='store_true',
                        help='Keep class label')

    parser.add_argument('--ohe-class', action='store_true',
                        help='Ohe class labels, only effective with --keep-class')

    parser.add_argument("--label-rules", type=str, nargs='*',
                        default=['vert-stroke', 'circle',
                                 'curve-left', 'curve-right',
                                 'hori-stroke'],
                        help='Rules to create multi labels')

    parser.add_argument("--rand-label-rules", type=int, nargs='?',
                        default=None,
                        help='Number of random label rules to create')

    parser.add_argument("--min-rand-num-digits-rule", type=int, nargs='?',
                        default=2,
                        help='Minumum number of random digits to create a random rule label rule')

    parser.add_argument("--max-rand-num-digits-rule", type=int, nargs='?',
                        default=5,
                        help='Maximum number of random digits to create a random rule label rule')

    parser.add_argument("-o", "--output", type=str, nargs='?',
                        default='./mlutils/datasets/mnist/',
                        help='output path')

    parser.add_argument("--bins", type=str, nargs='?',
                        default='blocks',
                        help='Histogram discretization technique')

    parser.add_argument("--remove-classes", type=int, nargs='+',
                        default=None,
                        help='Remove examples of certain classes')

    parser.add_argument("--masks", type=str, nargs='+',
                        # default=['up', 'down', 'left', 'right'],
                        default=[],
                        help='Specify which masked data to create')

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

    logging.info('******NOTE: this is a dirty script!\nBefore using it you should have\nlearned an autoencoder' +
                 ' with RAELK\n******')

    (orig_train_x, orig_train_y), (orig_valid_x, orig_valid_y), (orig_test_x,
                                                                 orig_test_y) = load_mnist_train_test_splits()

    n_cols = 28

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)

    orig_train_x_splits = []
    orig_valid_x_splits = []
    orig_test_x_splits = []
    if args.masks:
        logging.info('Creating masked versions for dataset!')

        for m in args.masks:

            masked_train_split_x = filter_image_data(orig_train_x, n_cols=n_cols, mask=m)
            orig_train_x_splits.append(masked_train_split_x)

            masked_valid_split_x = filter_image_data(orig_valid_x, n_cols=n_cols, mask=m)
            orig_valid_x_splits.append(masked_valid_split_x)

            masked_test_split_x = filter_image_data(orig_test_x, n_cols=n_cols, mask=m)
            orig_test_x_splits.append(masked_test_split_x)

            #
            # saving them
            masked_out_path = os.path.join(out_path,  m)
            save_mnist_split_raelk_format(masked_train_split_x, orig_train_y,
                                          masked_valid_split_x, orig_valid_y,
                                          masked_test_split_x, orig_test_y,
                                          output=masked_out_path)

    save_mnist_split_raelk_format(orig_train_x, orig_train_y,
                                  orig_valid_x, orig_valid_y,
                                  orig_test_x, orig_test_y,
                                  output=out_path)

    raelk_path = os.path.join(args.ae_dir, args.repr_file)
    data_path = os.path.join(out_path, args.data_file)

    train, valid, test = None, None, None

    if args.masks:
        repr_train_x_list = []
        repr_valid_x_list = []
        repr_test_x_list = []
        repr_train_y_list = []
        repr_valid_y_list = []
        repr_test_y_list = []
        #
        # load the various pieces up, then glue them together
        for m in args.masks:

            masked_raelk_path = os.path.join(args.ae_dir, m, args.repr_file)
            logging.info('Loading autoencoder repr from {}'.format(masked_raelk_path))
            masked_train, masked_valid, masked_test = \
                load_train_val_test_raelk_splits(masked_raelk_path, data_path)

            repr_train_x_list.append(masked_train[0])
            repr_train_y_list.append(masked_train[1])

            repr_valid_x_list.append(masked_valid[0])
            repr_valid_y_list.append(masked_valid[1])

            repr_test_x_list.append(masked_test[0])
            repr_test_y_list.append(masked_test[1])

        train_x = numpy.concatenate(repr_train_x_list, axis=1)
        valid_x = numpy.concatenate(repr_valid_x_list, axis=1)
        test_x = numpy.concatenate(repr_test_x_list, axis=1)

        print('Concatenated mask shapes train:{} valid:{} test:{}'.format(train_x.shape,
                                                                          valid_x.shape,
                                                                          test_x.shape))

        for j in range(1, len(args.masks)):
            assert_array_equal(repr_train_y_list[0], repr_train_y_list[j])
            assert_array_equal(repr_valid_y_list[0], repr_valid_y_list[j])
            assert_array_equal(repr_test_y_list[0], repr_test_y_list[j])

        train_y = repr_train_y_list[0]
        valid_y = repr_valid_y_list[0]
        test_y = repr_test_y_list[0]

    else:
        train, valid, test = load_train_val_test_raelk_splits(raelk_path, data_path)

        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

    if args.remove_classes:
        train_x, train_y = filter_samples_by_class(train_x, train_y, args.remove_classes)
        valid_x, valid_y = filter_samples_by_class(valid_x, valid_y, args.remove_classes)

    #
    # augmenting labels

    keep_original = args.keep_class
    one_hot_encoding_classes = args.ohe_class

    label_rule_funcs = None
    if args.rand_label_rules is not None:
        print('Generating {} random label rules'.format(args.rand_label_rules))
        label_rule_funcs = generate_random_label_rules(n_classes=10,
                                                       n_rules=args.rand_label_rules,
                                                       min_num_digits=args.min_rand_num_digits_rule,
                                                       max_num_digits=args.max_rand_num_digits_rule,
                                                       rand_gen=rand_gen)
    else:
        print('Retrieving label rules from dictionary!')
        label_rules = args.label_rules
        print('Considering rules ', label_rules)
        label_rule_funcs = [ENC_RULE_DICT[r] for r in label_rules]

    print('::::::::::::::::::\ndigit encodings')
    n_classes = 10
    for i in range(n_classes):
        enc = [r(i) for r in label_rule_funcs]
        print('{}\t{}'.format(i, ','.join(str(r_i) for r_i in enc)))

    aug_train_y = augment_labels(train_y,
                                 # label_rules=label_rules,
                                 label_rule_funcs=label_rule_funcs,
                                 keep_original=keep_original,
                                 one_hot_encoding_classes=False)
    aug_valid_y = augment_labels(valid_y,
                                 # label_rules=label_rules,
                                 label_rule_funcs=label_rule_funcs,
                                 keep_original=keep_original,
                                 one_hot_encoding_classes=False)
    aug_test_y = augment_labels(test_y,
                                # label_rules=label_rules,
                                label_rule_funcs=label_rule_funcs,
                                keep_original=keep_original,
                                one_hot_encoding_classes=False)

    print('\nAugmented train set labels {}'.format(aug_train_y.shape))
    print('Augmented valid set {}'.format(aug_valid_y.shape))
    print('Augmented test set {}'.format(aug_test_y.shape))

    #
    # concatenating X and Y
    n_x = train_x.shape[1]
    n_y = aug_train_y.shape[1]
    print('there are {} x and {} y'.format(n_x, n_y))
    feature_names_x = ['ae_{}'.format(i) for i in range(n_x)]
    feature_names_y = ['label_{}'.format(i) for i in range(n_y)]
    feature_names = feature_names_x + feature_names_y
    print('feature names ', feature_names)

    feature_types_x = ['continuous' for i in range(n_x)]
    feature_types_y = ['categorical' for i in range(n_y)]
    feature_types = feature_types_x + feature_types_y
    print('feature types', feature_types)

    #
    # numpy.array
    full_x = numpy.concatenate([train_x, valid_x, test_x], axis=0)
    full_y = numpy.concatenate([aug_train_y, aug_valid_y, aug_test_y], axis=0)
    print('fully dataset sizes', full_x.shape, full_y.shape)

    if args.stub_feature_info_file is not None:
        print('Reading domains by feature info file stub {}'.format(args.stub_feature_info_file))
        _fnames, _ftypes, _domains = load_feature_info_preprocess(args.stub_feature_info_file)
        domains_x = _domains[:n_x]
        print(domains_x)
        print(_domains)
    else:
        print('Estimating domains by embeddings')
        # domains_x = estimate_domains_range(full_x, feature_types_x)
        bin_method = args.bins
        bin_range_width = 0.001
        domains_x = [estimate_continuous_domain(full_x, i,
                                                range=(full_x[:, i].min() - bin_range_width,
                                                       full_x[:, i].max() + bin_range_width),
                                                binning_method=bin_method) for i in range(n_x)]
    domains_y = [estimate_categorical_domain(full_y, i) for i in range(n_y)]
    domains = domains_x + domains_y
    print('domains', domains)

    out_feature_info_path = os.path.join(out_path, 'aug.raelk.features')
    save_feature_info_dict(feature_names, feature_types,
                           domains, out_feature_info_path, range=False)
    print('Saved feature info file to ', out_feature_info_path)

    aug_train = numpy.concatenate([train_x, aug_train_y], axis=1)
    aug_valid = numpy.concatenate([valid_x, aug_valid_y], axis=1)
    aug_test = numpy.concatenate([test_x, aug_test_y], axis=1)

    #
    # saving modifyied dataset in text format

    out_train_path = os.path.join(out_path, 'aug.raelk.train.data')
    out_valid_path = os.path.join(out_path, 'aug.raelk.valid.data')
    out_test_path = os.path.join(out_path, 'aug.raelk.test.data')

    feature_formats = get_feature_formats(feature_types)
    numpy.savetxt(out_train_path, aug_train, fmt=feature_formats, delimiter=',')
    numpy.savetxt(out_valid_path, aug_valid, fmt=feature_formats, delimiter=',')
    numpy.savetxt(out_test_path, aug_test, fmt=feature_formats, delimiter=',')
    print('augmented splits dumped to:\n\t{}\n\t{}\n\t{}'.format(out_train_path,
                                                                 out_valid_path,
                                                                 out_test_path))

    #
    # load decoder
    # dec_path = '/home/valerio/Petto Redigi/trash/raelk.mnist_2017-05-22_15-00-18/ae.raelk.mnist.decoder.0'
    decoding = False
    if decoding:
        if args.masks:

            for i, m in enumerate(args.masks):
                print(m)
                masked_dec_path = os.path.join(args.ae_dir, m, 'ae.raelk.mnist.decoder.0')
                masked_decoder = load_ae_decoder(masked_dec_path)

                rec_train_x = decode_predictions(repr_train_x_list[i], masked_decoder)
                rec_valid_x = decode_predictions(repr_valid_x_list[i], masked_decoder)
                rec_test_x = decode_predictions(repr_test_x_list[i], masked_decoder)

                # plot_digit(orig_train_x[3])
                # plot_digit(rec_train_x[3], img_size=(28, 14))

        else:
            dec_path = os.path.join(args.ae_dir, 'ae.raelk.mnist.decoder.0')
            decoder = load_ae_decoder(dec_path)

            rec_train_x = decode_predictions(train_x, decoder)
            rec_valid_x = decode_predictions(valid_x, decoder)
            rec_test_x = decode_predictions(test_x, decoder)

            # plot_digit(orig_train_x[3])

            # plot_digit(rec_train_x[3])
