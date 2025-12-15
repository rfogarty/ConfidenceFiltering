
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns

def boxplotStat(filename,dataframe,stats,relabel=True,title=None) :
    pl.clf()
    dataframe.boxplot(stats)
    pl.title(stats)
    if relabel :
        indices=[r+1 for r in range(np.shape(dataframe.loc[:,stats])[1])]
        indiceslab=[r for r in range(np.shape(dataframe.loc[:,stats])[1])]
        labels=list(np.array(indiceslab).astype(str))
        indices.insert(0,0)
        labels.insert(0,'')
        pl.xticks(indices,labels)
    pl.ylim(0.5,1.0)
    if title is not None:
        pl.title(title)
    minorSpacing=MultipleLocator(0.05)
    majorSpacing=MultipleLocator(0.1)
    ax=pl.gca()
    ax.yaxis.set_minor_locator(minorSpacing)
    ax.yaxis.set_major_locator(majorSpacing)
    ax.grid(which='minor',axis='y',linestyle='--')
    #ax.grid(which='major',axis='y',linestyle='-',linewidth=2)
    pl.savefig(filename)

def lineplotStat(filename,dataframe,stats,relabel=True) :
    pl.clf()
    pl.plot(dataframe.loc[:,stats])
    pl.title(stats)
    #if relabel :
    #    indices=[r+1 for r in range(np.shape(dataframe.loc[:,stats])[1])]
    #    indiceslab=[r for r in range(np.shape(dataframe.loc[:,stats])[1])]
    #    labels=list(np.array(indiceslab).astype(str))
    #    indices.insert(0,0)
    #    labels.insert(0,'')
    #    pl.xticks(indices,labels)
    pl.savefig(filename)


def boxplotAccuracies(filename,dataframe) :
    pl.clf()
    dataframe.boxplot(['binary_accuracy','val_binary_accuracy'])
    pl.title('Accuracies')
    pl.savefig(filename)


def plotConfusionMatrix(name,confMat,ticklabels,title='Confusion Matrix') :
    pl.clf() 
    s=sns.heatmap(confMat.astype(int),cmap='Blues',yticklabels=ticklabels,xticklabels=ticklabels,annot=True,fmt='d',cbar=False) 
    pl.title(title)
    pl.xlabel('predicted') 
    pl.ylabel('truth') 
    pl.tight_layout() 
    pl.savefig(name) 


def plotMetric(plotfilename,metric,H,x=None,xlab="epoch",ylim=[-0.01,1.01]) :
    # plot the training loss and accuracy
    pl.style.use("ggplot")
    pl.figure()
    axes = pl.gca()
    axes.set_ylim(ylim)

    if 'History' in str(type(H)) :
        df = H.history
    else : # Assume it is a dictionary or Pandas DataFrame
        df = H

    if type(metric) is list :
        for met in metric :
            if x is None :
                N = len(df[met])
                xs = np.arange(0, N)
            else :
                xs = x
            pl.plot(xs, df[met], label=met)
        pl.title(f'Metric versus {xlab.capitalize()}')
    else :
        if x is None :
            N = len(df[metric])
            xs = np.arange(0, N)
        else :
            xs = x
        pl.plot(xs, df[metric], label=metric)
        pl.title(f'{metric} versus {xlab.capitalize()}')
        pl.ylabel(metric)
    pl.xlabel(xlab)
    pl.legend(loc='lower left')
    pl.savefig(plotfilename)


def plotMetricAndAux(plotfilename,metric,H,df,column) :
    # plot the training loss and accuracy
    pl.style.use("ggplot")
    fig, ax1 = pl.subplots()

    color1 = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('red', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    #pl.figure()
    #ax1 = pl.gca()
    ax1.set_ylim([-0.01,1.01])
    if type(metric) is list :
        for met in metric :
            N = len(H.history[met])
            ax1.plot(np.arange(0, N), H.history[met],color=color1,label=met)
        pl.title(f'Metrics versus Epoch')
    else :
        N = len(H.history[metric])
        pl.plot(np.arange(0, N), H.history[metric], color=color1,label=metric)
        pl.title(f'{metric} versus Epoch')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('blue', color=color2)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2)

    if type(column) in [list,tuple] :
        for col in columns :
            ser = df[col]
            N = len(ser)
            ax2.plot(np.arange(0, N),ser.tolist(),color=color2,label=str(col))
    else :
        ser = df[column]
        N = len(ser)
        ax2.plot(np.arange(0, N),ser.tolist(),color=color2,label=str(column))
 
    pl.xlabel("epoch")
    pl.legend(loc='lower left')
    pl.savefig(plotfilename)


def histograms(plotfilename,data,statistic,mapping) :
    pl.clf() 
    pl.figure()
    sns.histplot(data, x=statistic,hue=mapping,element="step",stat='probability',common_norm=False)
    pl.savefig(plotfilename)

