'''
Created on 24.02.2016

@author: alejomc
'''
from joblib.memory import Memory
import numpy
import os
import sys

from matplotlib import pyplot as plt, cm, colors
from matplotlib import rc
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as plticker
from tfspn.SPN import SPN, Splitting
from tfspn.measurements import getJointDist


os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'






rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

sys.setrecursionlimit(50000)

memory = Memory(cachedir="/tmp", verbose=0, compress=9)


def plotJointProb(filename, data, datarange):
    print(filename)
    print(data.shape)
    #spn = LearnSPN(alpha=0.001, min_instances_slice=30, cache=memory).fit_structure(data)
    spn = SPN.LearnStructure(data, min_instances_slice=30, row_split_method=Splitting.KmeansRows(), col_split_method=Splitting.RDCTest())
    
    print(spn)
    
    matplotlib.rcParams.update({'font.size': 16})
    pcm = cm.Greys


    f1 = 0
    f2 = 1

    x = data[:, f1]
    y = data[:, f2] 
    
    amin, amax = datarange[0], datarange[1]
    
    bins = numpy.asarray(list(range(amin, amax)))
    
    def getPxy(spn, f1, f2, xbins, ybins):
        import locale
        locale.setlocale(locale.LC_NUMERIC, 'C')
        Pxy = getJointDist(spn, f1, f2)

        jointDensity = numpy.zeros((max(xbins) + 1, max(ybins) + 1))
        for x  in xbins:
            for y in ybins:
                jointDensity[x, y] = Pxy(x, y)
        return jointDensity
    
   
    plt.clf()
    
    fig = plt.figure(figsize=(7, 7))
    
    
    # [left, bottom, width, height]
    xyHist = plt.axes([0.3, 0.3, 0.5, 0.5])
    cax = xyHist.imshow(getPxy(spn, f1, f2, bins, bins), extent=[amin, amax, amin, amax], interpolation='nearest', origin='lower', cmap=pcm)
    xyHist.set_xlim(amin, amax)
    xyHist.set_ylim(amin, amax)
    if amax > 20:
        xyHist.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
        xyHist.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
    
    xyHist.yaxis.grid(True, which='major', linestyle='-', color='darkgray')
    xyHist.xaxis.grid(True, which='major', linestyle='-', color='darkgray')
    
    xyHistcolor = plt.axes([0.82, 0.3, 0.03, 0.5])
    plt.colorbar(cax, cax=xyHistcolor)
    font = fm.FontProperties(size=32)
    # cax.yaxis.get_label().set_fontproperties(font)
    # cax.xaxis.get_label().set_fontproperties(font)
    
    xHist = plt.axes([0.05, 0.3, 0.15, 0.5])
    xHist.xaxis.set_major_formatter(NullFormatter())  # probs
    xHist.yaxis.set_major_formatter(NullFormatter())  # counts
    xHist.hist(x, bins=bins, orientation='horizontal', color='darkgray')
    xHist.invert_xaxis()
    xHist.set_ylim(amin, amax)
    
    yHist = plt.axes([0.3, 0.05, 0.5, 0.15])
    yHist.yaxis.set_major_formatter(NullFormatter())  # probs
    yHist.xaxis.set_major_formatter(NullFormatter())  # counts
    yHist.hist(y, bins=bins, color='darkgray')
    yHist.invert_yaxis()
    yHist.set_xlim(amin, amax)
    
    for elem in [xyHist, xHist, yHist]:
        elem.yaxis.grid(True, which='major', linestyle='-', color='darkgray')
        elem.xaxis.grid(True, which='major', linestyle='-', color='darkgray')

    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/" + filename, bbox_inches='tight', dpi=600)
    #0/0
    

numpy.random.seed(42)

gen = numpy.random.poisson(5, 1000)
gen2 = numpy.random.poisson(5, 1000)

genSorted = numpy.sort(gen)

genmixture = numpy.hstack((gen, numpy.random.poisson(25, len(gen))))
genmixture2 = numpy.hstack((gen2, numpy.random.poisson(25, len(gen2))))

genmixtureSorted = numpy.sort(genmixture)

negdependency = numpy.transpose(numpy.hstack((numpy.vstack(([0] * len(gen), gen)), numpy.vstack((gen, [0] * len(gen))))))

genmixture3 = numpy.hstack((numpy.random.poisson(2, 1000), numpy.random.poisson(13, 2000)))
negdependencymixture = numpy.transpose(numpy.hstack((numpy.vstack(([0] * len(genmixture3), genmixture3)), numpy.vstack((genmixture3, [0] * len(genmixture3))))))
negdependencymixture = negdependencymixture[~numpy.all(negdependencymixture == 0, axis=1)]

independent = numpy.transpose(numpy.vstack((gen, gen2)))
independentmixture = numpy.transpose(numpy.vstack((genmixture, genmixture2)))

opposite = numpy.transpose(numpy.vstack((genSorted, genSorted[::-1])))
oppositemixture = numpy.transpose(numpy.vstack((genmixtureSorted, genmixtureSorted[::-1])))

cooccur = numpy.transpose(numpy.vstack((gen, gen)))
cooccurmixture = numpy.transpose(numpy.vstack((genmixture, genmixture)))




plotJointProb("negdependency.pdf", negdependency, [0, 14])
# plotJointProb("negdependencymixture.pdf", negdependencymixture, [0, 20])

plotJointProb("independent.pdf", independent, [0, 14])
# plotJointProb("independentmixture.pdf", independentmixture, [0, 40])

plotJointProb("opposite.pdf", opposite, [0, 14])
plotJointProb("oppositemixture.pdf", oppositemixture, [0, 40])

plotJointProb("cooccur.pdf", cooccur, [0, 14])
plotJointProb("cooccurmixture.pdf", cooccurmixture, [0, 40])
    
