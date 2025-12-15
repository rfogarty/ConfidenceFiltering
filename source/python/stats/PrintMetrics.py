import numpy as np
import pandas as pd

def printPerfStats(perf) :
    print(f'Accuracy: {perf.accuracy}\nF1: {perf.F1}')
    print(f'Sensitivity: {perf.sensitivity}\nSpecificity: {perf.specificity}')
    print(f'Precision(PPV): {perf.PPV}\nNPV: {perf.NPV}')
    print(f'Confusion Matrix:')
    print( '                          Predicted')
    print( '                       GS3(N):   GS4(P):')
    print( '                   |----------------------')
    print(f'         / GS3(N): |   {perf.cm.TN}   |   {perf.cm.FP}')
    print( '  Truth <          |----------------------')
    print(f'         \\ GS4(P): |   {perf.cm.FN}   |   {perf.cm.TP}\n')


def printBoundedStats(perf) :
    print(f'Accuracy: {perf.accuracy.median} ({perf.accuracy.low},{perf.accuracy.high})')
    print(f'F1: {perf.F1.median} ({perf.F1.low},{perf.F1.high})')
    print(f'Sensitivity: {perf.sensitivity.median} ({perf.sensitivity.low},{perf.sensitivity.high})')
    print(f'Specificity: {perf.specificity.median} ({perf.specificity.low},{perf.specificity.high})')
    print(f'Precision(PPV): {perf.PPV.median} ({perf.PPV.low},{perf.PPV.high})')
    print(f'NPV: {perf.NPV.median} ({perf.NPV.low},{perf.NPV.high})')
    print(f'Confusion Matrix:')
    print( '                                    Predicted')
    print( '                       GS3(N):                       GS4(P):')
    print( '                   |------------------------------------------------------')
    print(f'         / GS3(N): |   {perf.cm.TN.median} ({perf.cm.TN.low},{perf.cm.TN.high})   |   {perf.cm.FP.median} ({perf.cm.FP.low},{perf.cm.FP.high})')
    print( '  Truth <          |----------------------')
    print(f'         \\ GS4(P): |   {perf.cm.FN.median} ({perf.cm.FN.low},{perf.cm.FN.high})   |   {perf.cm.TP.median} ({perf.cm.TP.low},{perf.cm.TP.high})\n')


def printAccuracyStats(dataframe) :
    dataframe.binary_accuracy.describe()


def printValAccuracyStats(dataframe) :
    dataframe.val_binary_accuracy.describe()


def printStats(dataframe,stats) :
    dataframe.loc[:,stats].describe()


# This function is intended to print accuracies for multi-class problems
# given two arrays for correct and incorrect values (per class). Additionally,
# the total accuracy is also computed.
def printAccuracies(correct,incorrect) :
    totcorrect = 0
    totincorrect = 0
    for idx,(c,i) in enumerate(zip(correct,incorrect)) :
        totcorrect += c
        totincorrect += i
        if ((c + i) > 0) : acc = c / (c + i)
        else : acc = 'n/a'
        print(f"Digit {idx} accuracy : {acc} ({c},{i})")
    if ((totcorrect + totincorrect) > 0) : acc = totcorrect / (totcorrect + totincorrect)
    else : acc = 'n/a'
    print(f"Total accuracy : {acc}")
    return (acc,totcorrect,totincorrect)


def isEven(N) :
    return int(N/2)*2 == N

def printBounds(name,m,usePandas=False) :

    #breakpoint()
    if type(m) is list :
        m = np.array(m)
    m = np.sort(m)
    if usePandas :
        df = pd.DataFrame({name:m})
        print(df.describe())
    else :
        length=m.shape[0]
        i2p5=int(0.025*length)-1
        i5=int(0.05*length)-1
        i25=int(0.25*length)-1
        i50=int(0.5*length)-1
        i75=int(0.75*length)-1
        i95=int(0.95*length)-1
        i97p5=int(0.975*length)-1
        if isEven(length) :
            # Only 50% (median) is compensated when N is even
            #print(f'{name}: {m[0]}:low,{m[i2p5]}:2.5%,{m[i5]}:5%,{m[i25]}:25%,{(m[i50]+m[i50+1])/2}:50%,{m[i75]}:75%,{m[i95]}:95%,{m[i97p5]}:97.5%,{m[length-1]}:high,{m.mean()}:mean,{m.var()}:var')
            print(f'{name}: {m.mean():.3f} ({m[i2p5]:.3f},{m[i97p5]:.3f})')
        else :
            #print(f'{name}: {m[0]}:low,{m[i2p5]}:2.5%,{m[i5]}:5%,{m[i25]}:25%,{m[i50]}:50%,{m[i75]}:75%,{m[i95]}:95%,{m[i97p5]}:97.5%,{m[length-1]}:high,{m.mean()}:mean,{m.var()}:var')
            print(f'{name}: {m.mean():.3f} ({m[i2p5]:.3f},{m[i97p5]:.3f})')


