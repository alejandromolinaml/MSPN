'''
Created on 13.06.2016

@author: alejomc
'''
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import xlim
import numpy

import matplotlib.pyplot as plt
import matplotlib.ticker as tick


def plotBoxes(stats, measure, nrows=4, ncols=1, fs = 10, size=(6,8)):
    
    plt.figure()
    plt.clf()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)
    fig.suptitle(measure)
    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            stat = stats[idx]
            idx += 1
            data = numpy.transpose(numpy.asarray([stat.getValues(method, measure) for method in stat.getMethods(measure)]))
            if nrows > 1 and ncols > 1:
                ax = axes[i,j]
            elif nrows == 1:
                ax = axes[j]
            elif ncols == 1:
                ax = axes[i]
                
            print(data)
            print(data.shape)
            ax.boxplot(data, labels=list(stat.getMethods(measure)), vert=False, widths=0.8)
            #ax.set_ylabel(stat.name, fontsize=fs+4)
            #ax.set_ylabel("Left Y-Axis Data")
            ax.text(1.05, 0.5, stat.name,
                horizontalalignment='center',
                verticalalignment='center',
                rotation='vertical',
                fontsize=fs+8,
                transform=ax.transAxes)
            ax.tick_params(labelsize=fs)
            ax.tick_params(axis='x', which='major', pad=2)
            xlim(0,10)
            #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            #ax.xaxis.set_major_formatter(tick.FuncFormatter(lambda x,y: '{:1.1e}'.format(x).replace('+0', '')))
            #for l in ax.get_yticklabels():
            #    l.set_fontsize(fs+4)

#    for ax in axes.flatten():
#        ax.set_yscale('log')
#        ax.set_yticklabels([])

    #fig.subplots_adjust(hspace=0.2)
    #plt.subplots_adjust(left=0.9, right=0.1, top=0.9, bottom=0.1)
    
    #plt.tight_layout()



def plotStats(stats, fname):
    pp = PdfPages(fname)
    
    for measure in stats.getMeasures():
        dt = []
        names = []
        plt.clf()
        for method in stats.getMethods(measure):
            names.append(method)
            print(method, measure)
            print(stats.getValues(method, measure))
            vals = stats.getValues(method, measure)
            #vals = reject_outliers(numpy.asarray(vals),2)
            print(vals)
            dt.append(vals)


        #plt.figure(figsize=(6,4))
        plt.title(measure, weight='extra bold')
    
        plt.boxplot(dt)
        axes = plt.gca()
        
        alldt = numpy.asarray(sum(dt,[]))
        #alldt = reject_outliers(alldt)
        
        mn = min(alldt)
        mx = max(alldt)
        
        #axes.set_ylim([min(alldt),max(alldt)])
        axes.set_ylim([int(mn),int(mx)])
        plt.xticks(range(1,len(names)+1),names, weight='bold')
    
        pp.savefig(plt.gcf())
    pp.close()
    
def reject_outliers(data, m=3):
    #this is a hack
    return data[abs(data - numpy.median(data)) < m * numpy.std(data)].tolist()