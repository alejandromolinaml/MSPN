import os
import itertools

import numpy

rand_gen = numpy.random.RandomState(1337)

LEAVES = ['piecewise', 'isotonic', 'histogram']
N_TRIALS = 10
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
          101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
assert len(PRIMES) >= N_TRIALS
# SEEDS = rand_gen.choice(numpy.arange(100000), replace=False, size=N_TRIALS)
SEEDS = PRIMES[:N_TRIALS]

N_RAND_LABELS = [
    2,
    4,
    8,
    16,
    32,
    64
]

PYTHON_BIN = "ipython -- "

AUGMENT_MNIST_PATH = " experiments/mnist/augmenting_mnist.py "
LEARNSPN_PATH = " bin/learnspn.py "
VIS_PATH = " bin/mnist_vis.py "

# AUTOENCODER_PATH = " ./experiments/mnist/16/raelk.mnist_2017-05-26_11-58-03/ "
AUTOENCODER_PATH = " ./mlutils/ae "

OUTPUT = "exp/mnist-priv-info"

MIN_RAND_DIGITS = 2
MAX_RAND_DIGITS = 5

RDC_THRESHOLD = 0.15

MIN_INST_SLICES = 200

ALPHA = 1

EMB_FEATURES = 16

augment_cmd_list = []
learn_cmd_list = []
inference_cmd_list = []

for leaf in LEAVES:

    for n_rand_labels in N_RAND_LABELS:

        exp_id = 0
        mem_map_base_path = os.path.join(OUTPUT, leaf, 'rand-{}'.format(n_rand_labels))
        os.makedirs(mem_map_base_path, exist_ok=True)
        mem_map_path = os.path.join(mem_map_base_path, 'mem.map')

        for seed in SEEDS:

            # print('')
            #
            # create new dataset
            augment_cmd = PYTHON_BIN
            augment_cmd += AUGMENT_MNIST_PATH
            augment_cmd += " --ae-dir {}".format(AUTOENCODER_PATH)
            output_path = os.path.join(OUTPUT, leaf, 'rand-{}'.format(n_rand_labels), str(seed))
            os.makedirs(output_path, exist_ok=True)
            augment_cmd += " -o {} ".format(output_path)
            augment_cmd += " --seed {} ".format(seed)
            augment_cmd += " --rand-label-rules {} ".format(n_rand_labels)
            augment_cmd += " --min-rand-num-digits-rule {} ".format(MIN_RAND_DIGITS)
            augment_cmd += " --max-rand-num-digits-rule {} ".format(MAX_RAND_DIGITS)
            augment_cmd += "--keep-class --bins auto"

            # print(augment_cmd)
            augment_cmd_list.append(augment_cmd)

            #
            # learn SPN on dataset
            learn_cmd = PYTHON_BIN
            learn_cmd += LEARNSPN_PATH
            learn_cmd += ' aug.raelk '
            learn_cmd += ' --data-dir {} '.format(output_path)
            learn_cmd += '   -v 2    --seed 1337 '
            learn_cmd += ' --row-split rdc-kmeans --row-split-args "ohe=0,n_clusters=2" '
            learn_cmd += ' --col-split rdc --col-split-args "threshold={},ohe=0,linear=1" '.format(
                RDC_THRESHOLD)
            learn_cmd += ' --min-inst-slice {} '.format(MIN_INST_SLICES)
            learn_cmd += ' --alpha {} '.format(ALPHA)
            learn_cmd += ' --leaf {} '.format(leaf)
            spn_output_path = os.path.join(output_path, 'spn')
            os.makedirs(spn_output_path, exist_ok=True)
            learn_cmd += ' -o {} '.format(spn_output_path)

            # print(learn_cmd)
            learn_cmd_list.append(learn_cmd)

            #
            # make inference
            inference_cmd = PYTHON_BIN
            inference_cmd += VIS_PATH
            inference_cmd += " aug.raelk "
            inference_cmd += " --exp-id {} ".format(exp_id)
            inference_cmd += ' --memmap {} '.format(mem_map_path)
            inference_cmd += " --data-dir {} ".format(output_path)
            inference_cmd += " -v 2 "
            spn_model_path = os.path.join(spn_output_path, 'aug.raelk', 'None', 'spn.0.pklz')
            inference_cmd += " --spn {} ".format(spn_model_path)
            inference_cmd += " --emb-features {} ".format(EMB_FEATURES)
            inference_cmd += " --aug-features {} ".format(n_rand_labels + 1)
            inference_cmd += " --ae-path {} ".format(AUTOENCODER_PATH)
            inference_cmd += "  -o {}".format(output_path)
            inference_cmd += " --privileged-inference"

            # print(inference_cmd)
            inference_cmd_list.append(inference_cmd)

            exp_id += 1

        n_cols = 12
        mem_map = numpy.memmap(mem_map_path, dtype='float', mode='w+', shape=(exp_id, n_cols))

print('')
for augment_cmd in augment_cmd_list:
    print(augment_cmd)

print('')
for learn_cmd in learn_cmd_list:
    print(learn_cmd)

print('')
for inference_cmd in inference_cmd_list:
    print(inference_cmd)
