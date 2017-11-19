import argparse

import numpy

BASE_ACCS = {'isotonic': 0.8915,
             'histogram': 0.9033,
             'piecewise': 0.8913}

BASE_LLS = {'isotonic': 2.322526819,
            'histogram': 2.99994659,
            'piecewise': 2.319379366}


def best_ll_exps(fp, target_col, res_col):

    best_id = numpy.argmax(fp[:, target_col])
    exp_id = int(fp[best_id, 0])
    print('Best column id ',  best_id, 'exp id', exp_id)
    print(fp[:, target_col])
    print(fp[best_id])
    test_ll = fp[best_id, res_col]
    print('\t\ttest ll', test_ll)

    return test_ll, exp_id


def relative_improvement(values, baseline):
    return (values - baseline) / baseline * 100

N_TRIALS = 10
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
          101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
assert len(PRIMES) >= N_TRIALS
# SEEDS = rand_gen.choice(numpy.arange(100000), replace=False, size=N_TRIALS)
SEEDS = PRIMES[:N_TRIALS]


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='(MLC) dataset name')


#
# parsing the args
args = parser.parse_args()

#
# isotonic/rand-2
dataset = args.dataset
memmap_path = '/media/valerio/formalità/mnist-priv-info/{}/mem.map'.format(dataset)

baseline_ll = None
baseline_acc = None
if 'piecewise' in dataset:
    baseline_ll = BASE_LLS['piecewise']
    baseline_acc = BASE_ACCS['piecewise']
elif 'histogram' in dataset:
    baseline_ll = BASE_LLS['histogram']
    baseline_acc = BASE_ACCS['histogram']
elif 'isotonic' in dataset:
    baseline_ll = BASE_LLS['isotonic']
    baseline_acc = BASE_ACCS['isotonic']
else:
    raise ValueError('ERROR')

exp_cmd_example = '/home/valerio/Downloads/mnist.exp'

# cmd_lines = None
# with open(exp_cmd_example, 'r') as f:
#     cmd_lines = f.readlines()

# cmd_lines = [l for l in cmd_lines if 'bin/learnspn.py  {} '.format(dataset) in l]

fp = numpy.memmap(memmap_path, dtype='float', mode='r+')
fp = fp.reshape(-1, 12)

n_exps = fp.shape[0]
print('There are {} experiments'.format(n_exps))

# assert len(cmd_lines) == n_exps


train_avg_lls = fp[:, 1]
valid_avg_lls = fp[:, 2]
test_avg_lls = fp[:, 3]
marg_train_avg_lls = fp[:, 4]
marg_valid_avg_lls = fp[:, 5]
marg_test_avg_lls = fp[:, 6]
mpe_class_accuracys = fp[:, 7]
map_class_accuracys = fp[:, 8]
mpe_labels_jacs = fp[:, 9]
mpe_labels_hams = fp[:, 10]
mpe_labels_exas = fp[:, 11]


avg_train_avg_ll = train_avg_lls.mean()
avg_valid_avg_ll = valid_avg_lls.mean()
avg_test_avg_ll = test_avg_lls.mean()
avg_marg_train_avg_ll = marg_train_avg_lls.mean()
avg_marg_valid_avg_ll = marg_valid_avg_lls.mean()
avg_marg_test_avg_ll = marg_test_avg_lls.mean()
avg_mpe_class_accuracy = mpe_class_accuracys.mean()
avg_map_class_accuracy = map_class_accuracys.mean()
avg_mpe_labels_jac = mpe_labels_jacs.mean()
avg_mpe_labels_ham = mpe_labels_hams.mean()
avg_mpe_labels_exa = mpe_labels_exas.mean()

n_failures = (fp.sum(axis=1) == 0).sum()
print('There are {} not run experiments'.format(n_failures))

print('baseline test marg ll:\t{}'.format(baseline_ll))
print('baseline test class acc:\t{}'.format(baseline_acc))
print('')
print('avg')
print('\t'.join(str(f) for f in [avg_train_avg_ll,
                                 avg_valid_avg_ll,
                                 avg_test_avg_ll,
                                 avg_marg_train_avg_ll,
                                 avg_marg_valid_avg_ll,
                                 avg_marg_test_avg_ll,
                                 avg_mpe_class_accuracy,
                                 avg_map_class_accuracy,
                                 avg_mpe_labels_jac,
                                 avg_mpe_labels_ham,
                                 avg_mpe_labels_exa]))

imp_marg_test_lls = relative_improvement(marg_test_avg_lls, baseline_ll)
print('relative marg test lls:\n\t{}\t{}\t{}'.format('\t'.join(str(t) for t in imp_marg_test_lls),
                                                     imp_marg_test_lls.mean(),
                                                     imp_marg_test_lls.std()))
imp_mpe_acc_test_lls = relative_improvement(mpe_class_accuracys * 100, baseline_acc * 100)
print('relative mpe test class acc:\n\t{}\t{}\t{} '.format('\t'.join(str(t)
                                                                     for t in imp_mpe_acc_test_lls),
                                                           imp_mpe_acc_test_lls.mean(),
                                                           imp_mpe_acc_test_lls.std()))
imp_map_acc_test_lls = relative_improvement(map_class_accuracys * 100, baseline_acc * 100)
print('relative map test class acc:\n\t{}\t{}\t{}'.format('\t'.join(str(t)
                                                                    for t in imp_map_acc_test_lls),
                                                          imp_map_acc_test_lls.mean(),
                                                          imp_map_acc_test_lls.std()))
