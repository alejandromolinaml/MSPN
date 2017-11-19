import os
import itertools

import numpy

from mlutils.datasets import loadMLC


def cartesian_product_dict_list(d):
    return (dict(zip(d, x)) for x in itertools.product(*d.values()))

BASE_PATH = os.path.dirname(__file__)
DATA_DIR = 'mlutils/datasets/MLC/proc-db/proc/'

PYTHON_INTERPRETER = 'ipython -- '
# PYTHON_INTERPRETER = 'python3 '
CMD_LINE_BIN_PATH = ' bin/learnspn.py '
VERBOSITY_LEVEL = ' -v 2 '
SEED = 1337

# BINNING_METHODS = [
#     "blocks",
#     "unique",
#     "auto"
# ]

BINNING_METHODS = [
    "cat",
    # "unique"
]


DATASETS = [
    "anneal-U",
    "australian",
    "auto",
    "balance-scale",
    "breast",
    "breast-cancer",
    "cars",
    "cleve",
    "crx",
    "diabetes",
    "german",
    "german-org",
    "glass",
    "glass2",
    "heart",
    "iris"
]

# ROW_SPLIT_METHODS = {
#     'kmeans': {'n_clusters': [2], 'ohe': [0, 1]},
#     'rdc-kmeans': {'n_clusters': [2], 'k': [1, 10, 100], 'ohe': [0, 1]},
#     'rand-cond': {},
#     'rand-mode-split': {},
#     'rand-split': {},
#     'gower': {},
#     #'dbscan': {'eps': [0.01, 0.1, 0.2, 0.3]}
# }

ROW_SPLIT_METHODS = {
    'rdc-kmeans': {'n_clusters': [2], 'k': [20], 'ohe': [1]},
    'gower': {'n_clusters': [2]},
}


# COL_SPLIT_METHODS = {'rdc': {'threshold': [0.1, 0.2, 0.3, 0.7, 0.8, 0.9], 'ohe': [0, 1], 'linear': [0, 1]},
#                      'gdt': {'threshold': [0.001, 0.01, 0.1, 0.15, 0.20]}}

COL_SPLIT_METHODS = {'rdc': {'threshold': [0.1, 0.2, 0.3], 'ohe': [1], 'linear': [1]},
                     }

MIN_INST_SLICES_PERC = [0.05, 0.1, 0.2]

MIN_INST_SLICES = [50, 100, 150]

ALPHAS = [0.0, 1.0]

# ISOTONIC = [True, False]

LEAVES = [
    'baseline',
    'histogram', 'piecewise', 'isotonic'
]

PRIOR_WEIGHTS = [0.1, 0.01, 0.001]

# N_BOOTSTRAPS = [None, 5]

# AVERAGE_BOOTSTRAPS = [False, True]

# PRODUCT_FIRST = [False, True]

OUTPUT_DIR = './exp/learnspn/'


#
# opening the memmap


for dataset in DATASETS:

    for bins in BINNING_METHODS:

        exp_id = 0

        (train, valid, test), _f_names, _f_types, _f_domains = loadMLC(dataset,
                                                                       base_path=DATA_DIR,
                                                                       data_dir=bins)
        data_dir = os.path.join(DATA_DIR, bins)
        output_dir = os.path.join(OUTPUT_DIR, bins)

        # mem_map_base_path = os.path.join(BASE_PATH, dataset, bins)
        mem_map_base_path = os.path.join(output_dir, dataset)
        mem_map_path = os.path.join(mem_map_base_path, 'mem.map')
        os.makedirs(mem_map_base_path, exist_ok=True)

        configurations = itertools.product(ROW_SPLIT_METHODS,
                                           COL_SPLIT_METHODS,
                                           MIN_INST_SLICES,
                                           ALPHAS,
                                           LEAVES,
                                           PRIOR_WEIGHTS
                                           # ISOTONIC,
                                           # N_BOOTSTRAPS,
                                           # AVERAGE_BOOTSTRAPS,
                                           # PRODUCT_FIRST
                                           )

        for r_split, c_split,  \
                min_inst, alpha, \
                leaf, prior_weight in configurations:
                # isotonic, n_bootstraps, \
                # avg_bootstraps, product_first in configurations:

            r_split_args = ROW_SPLIT_METHODS[r_split]
            c_split_args = COL_SPLIT_METHODS[c_split]

            # min_inst_number = int(min_inst * train.shape[0])
            min_inst_number = min_inst

            for r_args in cartesian_product_dict_list(r_split_args):

                r_arglist_str = ','.join('{}={}'.format(k, v) for k, v in r_args.items())
                r_arg_str = ' --row-split-args "{}"'.format(
                    r_arglist_str) if len(r_arglist_str) > 1 else ''

                for c_args in cartesian_product_dict_list(c_split_args):

                    c_arglist_str = ','.join('{}={}'.format(k, v) for k, v in c_args.items())
                    c_arg_str = ' --col-split-args "{}"'.format(
                        c_arglist_str) if c_arglist_str else ''

                    cmd_line_str = PYTHON_INTERPRETER
                    cmd_line_str += CMD_LINE_BIN_PATH
                    cmd_line_str += ' {} '.format(dataset)
                    cmd_line_str += ' {} '.format(VERBOSITY_LEVEL)
                    cmd_line_str += ' --data-dir {} '.format(data_dir)
                    cmd_line_str += ' --output {} '.format(output_dir)
                    cmd_line_str += ' --exp-id {} '.format(exp_id)
                    cmd_line_str += ' --memmap {} '.format(mem_map_path)
                    cmd_line_str += ' --seed {} '.format(SEED)
                    cmd_line_str += ' --row-split {} '.format(r_split)
                    cmd_line_str += ' {} '.format(r_arg_str)
                    cmd_line_str += ' --col-split {} '.format(c_split)
                    cmd_line_str += ' {} '.format(c_arg_str)
                    cmd_line_str += ' --min-inst-slice {} '.format(min_inst_number)
                    cmd_line_str += ' --alpha {} '.format(alpha)
                    cmd_line_str += ' --leaf {} '.format(leaf)
                    cmd_line_str += ' --prior-weight {} '.format(prior_weight)
                    # if isotonic:
                    #     cmd_line_str += ' --isotonic '
                    # if n_bootstraps is not None:
                    #     cmd_line_str += ' --bootstraps {} '.format(n_bootstraps)
                    # if avg_bootstraps:
                    #     cmd_line_str += ' --average-bootstraps '
                    # if product_first:
                    #     cmd_line_str += ' --product-first '

                    print(cmd_line_str)

                    exp_id += 1

        n_cols = 13
        mem_map = numpy.memmap(mem_map_path, dtype='float', mode='w+', shape=(exp_id, n_cols))
