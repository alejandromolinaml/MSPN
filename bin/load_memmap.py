import argparse

import numpy


def best_ll_exps(fp, target_col, res_col):

    best_id = numpy.argmax(fp[:, target_col])
    exp_id = int(fp[best_id, 0])
    print('Best column id ',  best_id, 'exp id', exp_id)
    print(fp[:, target_col])
    print(fp[best_id])
    test_ll = fp[best_id, res_col]
    print('\t\ttest ll', test_ll)

    return test_ll, exp_id

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='(MLC) dataset name')

#
# parsing the args
args = parser.parse_args()

dataset = args.dataset
memmap_path = '/home/valerio/Downloads/unique/{}/mem.map'.format(dataset)

exp_cmd_example = '/home/valerio/Downloads/experiments.txt'

cmd_lines = None
with open(exp_cmd_example, 'r') as f:
    cmd_lines = f.readlines()

cmd_lines = [l for l in cmd_lines if 'bin/learnspn.py  {} '.format(dataset) in l]

fp = numpy.memmap(memmap_path, dtype='float', mode='r+')
fp = fp.reshape(-1, 13)

n_exps = fp.shape[0]
print('There are {} experiments'.format(n_exps))

assert len(cmd_lines) == n_exps

n_half = n_exps // 2

print('There are {} experiments with gower  and rdc_kmeans'.format(n_half))
rdc_kmeans_ids = numpy.arange(n_half)
gower_ids = numpy.arange(n_half, n_exps)

rdc_kmeans_exps = fp[rdc_kmeans_ids]
gower_exps = fp[gower_ids]

gower_cmd_ids = numpy.array([i for i, line in enumerate(cmd_lines) if '--row-split gower' in line])
rdc_cmd_ids = numpy.array(
    [i for i, line in enumerate(cmd_lines) if '--row-split rdc-kmeans' in line])

print('GOWER IDS')
print(gower_cmd_ids)

print('RDC IDS')
print(rdc_cmd_ids)


rdc_kmeans_exps = fp[rdc_cmd_ids]
gower_exps = fp[gower_cmd_ids]

iso_ids = numpy.array([i for i, line in enumerate(cmd_lines) if '--leaf isotonic' in line])
hist_ids = numpy.array([i for i, line in enumerate(cmd_lines) if '--leaf histogram' in line])
pw_ids = numpy.array([i for i, line in enumerate(cmd_lines) if '--leaf piecewise' in line])

print(iso_ids)
print(hist_ids)
print(pw_ids)

#
# intersecting
gower_iso_ids = numpy.intersect1d(iso_ids, gower_cmd_ids)
gower_hist_ids = numpy.intersect1d(hist_ids, gower_cmd_ids)
gower_pw_ids = numpy.intersect1d(pw_ids, gower_cmd_ids)

rdc_iso_ids = numpy.intersect1d(iso_ids, rdc_cmd_ids)
rdc_hist_ids = numpy.intersect1d(hist_ids, rdc_cmd_ids)
rdc_pw_ids = numpy.intersect1d(pw_ids, rdc_cmd_ids)

print('GOWER iso ids')
print(gower_iso_ids)
print('GOWER hist ids')
print(gower_hist_ids)
print('GOWER pw ids')
print(gower_pw_ids)


print('RDC iso ids')
print(rdc_iso_ids)
print('RDC hist ids')
print(rdc_hist_ids)
print('RDC pw ids')
print(rdc_pw_ids)

valid_feature = 3
test_feature = 4
print('GOWER HIST')
gh, gh_exp_id = best_ll_exps(fp[gower_hist_ids], valid_feature, test_feature)
print(cmd_lines[gh_exp_id])
print('GOWER PW')
gp, gp_exp_id = best_ll_exps(fp[gower_pw_ids], valid_feature, test_feature)
print(cmd_lines[gp_exp_id])
print('GOWER ISO')
gi, gi_exp_id = best_ll_exps(fp[gower_iso_ids], valid_feature, test_feature)
print(cmd_lines[gi_exp_id])

print('RDC HIST')
rh, rh_exp_id = best_ll_exps(fp[rdc_hist_ids], valid_feature, test_feature)
print(cmd_lines[rh_exp_id])
print('RDC PW')
rp, rp_exp_id = best_ll_exps(fp[rdc_pw_ids], valid_feature, test_feature)
print(cmd_lines[rp_exp_id])
print('RDC ISO')
ri, ri_exp_id = best_ll_exps(fp[rdc_iso_ids], valid_feature, test_feature)
print(cmd_lines[ri_exp_id])

n_failures = (fp.sum(axis=1) == 0).sum()
print('There are {} not run experiments'.format(n_failures))

print()
print('\t'.join(str(k) for k in [gh, gp, gi, rh, rp, ri]))
