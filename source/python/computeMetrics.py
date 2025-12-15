import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from dataclasses import dataclass
from stats.ComputeMetrics import computeStats,computeBinomialMetrics
from stats.PrintMetrics import printPerfStats
from stats.MetricTypes import ConfMatrix
from stats.MetricPlots import histograms
import argparse
from stats.PrintMetrics import printBounds
#from parseResultMetrics import *
import parseResultMetrics as parser


def addArguments(argparser) :
    argparser.add_argument('-p','--prefix',dest='prefix',type=str,required=False,help='Prefix of CSV file to parse',metavar='PREF',default='Results_')
    parser.addParseResultsArguments(argparser)


def processCommandline(rawArgs=None) :
    argparser = argparse.ArgumentParser(description='Compute metrics from Result (CSV) files')
    addArguments(argparser)
    args=argparser.parse_args(rawArgs)
    parser.processParseResultsArguments(args)
    return args


def compileResults(args,returnConfidence=False) :
    accuracies = list()
    f1s = list()
    aucs = list()
    confidences = list()
    lowestConfidences = list()
    for df,testn,_,_ in parser.parseResults(args.prefix,args) :
        #breakpoint()
        acc,f1,auc,lowconf=computeBinomialMetrics(df['Outcome'],df['Prediction'],df['Truth'],numModels=1,modelNumber=testn,confidences=df['Confidence'].to_numpy())
        accuracies.extend(acc)
        f1s.extend(f1)
        aucs.extend(auc)
        # Note Confidences will be an array of arrays...
        confidences.append(df['Confidence'].to_numpy())
        lowestConfidences.extend(lowconf)
        if 'RANSAC_Thresh' in df.columns:
            dfRANSAC = df[['PatchName','RANSAC_Thresh']]
            if not os.path.exists(f'InferencePerFileTest{testn}.txt') :
                print(f'Generating InferencePerFileTest{testn}.txt')
                dfRANSAC.to_csv(f'InferencePerFileTest{testn}.txt',sep=':',index=False)
            else :
                print(f'WARNING: InferencePerFileTest{testn}.txt exists, so not regenerating')
    
    printBounds('Accuracy',accuracies)
    printBounds('F1',f1s)
    printBounds('AUC',aucs)
    
    print(aucs)

    # This is extremely sketch but will isolate other code that uses this for now.
    if returnConfidence :
        return (accuracies,f1s,aucs,confidences,lowestConfidences)
    else :
        return (accuracies,f1s,aucs)


if __name__ == "__main__" :
    args = processCommandline()
    compileResults(args)


#dfNeg = df[df['Truth'] == 0]
#dfNeg = dfNeg.reset_index(drop=True)
#TN = dfNeg[dfNeg['Outcome'] == 0].shape[0]
#FP = dfNeg.shape[0] - TN
#dfPos = df[df['Truth'] == 1]
#dfPos = dfPos.reset_index(drop=True)
#TP = dfPos[dfPos['Outcome'] == 1].shape[0]
#FN = dfPos.shape[0] - TP
#
#perf = computeStats(ConfMatrix(TP,TN,FP,FN))
#printPerfStats(perf)

#suffix=''
#if args.ensemble :
#    suffix='Ensemble'
#histograms(f'PredictionsTest{testn}{suffix}.png',df,'Prediction','Truth')

