'''
    Each line is a sequence.
    Each itemset is a set of items, where items are positive integers separated
    by spaces.
    "-1"  indicates the end of an itemset.
    "-2" indicates the end of a sequence.
'''

from collections import Counter
import os
import sys

import numpy as np


def parse_spmf_format(data_str):
    itemsets = []
    max_val = 0
    min_val = sys.maxsize
    for i, line in enumerate(data_str.split('-2')):
        itemset = line.strip('-2').strip().split("-1")
        itemset = list(filter(lambda x: x.strip() != '', itemset))
        if len(itemset) > 0:
            itemset = map(int, itemset)
            cnt = Counter(itemset)
            max_val = max(max_val, max(cnt.keys()))
            min_val = min(min_val, min(cnt.keys()))
            itemsets.append(cnt)

    # print len(itemsets)
    # print max_val, min_val
    data = np.zeros((len(itemsets), max_val - min_val + 1))
    for i, itemset in enumerate(itemsets):
        for item, count in itemset.items():
            data[i, item - 1] = count
    return data

if __name__ == '__main__':
    fn_data = "MSNBC.txt"
    with open(fn_data) as fp:
        data_str = fp.read()

    data = parse_spmf_format(data_str)

    header = ",".join("frontpage news tech local opinion on_air misc weather msn_news health living business msn_sports sports summary bbs travel".split())
    # print header
    np.savetxt("MSNBC.pdn.csv", data, fmt='%d', delimiter=',', header=header, comments='')
