import os

import numpy

from mlutils.datasets import loadMLC


OUTPUT_DIR = 'exp/mutualinfo/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

mem_map_path = os.path.join(OUTPUT_DIR, 'mem.map')


((train, valid, test), feature_names, feature_types, domains) = loadMLC("autism", data_dir="datasets/autism/proc/unique")

nfeatures = len(feature_types)

mem_map = numpy.memmap(mem_map_path, dtype='float', mode='w+', shape=(nfeatures, nfeatures))

for i in range(nfeatures):
    for j in range(nfeatures):
        if j <= i:
            continue
        
        print("python3 experiments/mutualinformation/mitest.py '%s' %s %s" % (mem_map_path, i, j))
