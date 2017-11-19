from collections import OrderedDict


def numpytoordereddict(data, colnames):
    hdata = OrderedDict()
    for c in range(data.shape[1]):
        hdata[colnames[c]] = list(data[:, c])
        
    return hdata