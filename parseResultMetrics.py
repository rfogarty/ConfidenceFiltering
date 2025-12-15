import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
import bisect
from stats.ComputeMetrics import computeStats,computeBinomialMetrics
from stats.PrintMetrics import printPerfStats
from stats.MetricTypes import ConfMatrix
from stats.MetricPlots import histograms
import argparse
from stats.PrintMetrics import printBounds
from data import readLabels,stripPathnames


def addParseResultsArguments(argparser) :
    argparser.add_argument('--ensemble',action=argparse.BooleanOptionalAction,default=False,help='Whether to perform ensemble scoring')
    argparser.add_argument('--smartEnsemble',action=argparse.BooleanOptionalAction,default=False,help='Whether to perform smart ensemble scoring')
    argparser.add_argument('--model_1st_ensemble',action=argparse.BooleanOptionalAction,default=False,help='Whether to perform model 1st ensemble scoring')
    argparser.add_argument('--split_1st_ensemble',action=argparse.BooleanOptionalAction,default=False,help='Whether to perform split 1st ensemble scoring')
    argparser.add_argument('--ransacScore',action=argparse.BooleanOptionalAction,default=False,help='Whether to perform ransac ensemble scoring across all Test/Splits/Models')
    argparser.add_argument('-T','--testn', dest='testn',metavar='TEST', type=int,action='append', nargs='+',help='test number',required=True)
    argparser.add_argument('-M','--modeln', dest='modeln',metavar='MODEL', type=int,action='append', nargs='+',help='model number(s)',required=False,default=[])
    argparser.add_argument('-S','--splitn',dest='splitn', metavar='SPLIT', type=int,action='append', nargs='+',help='splits',required=True)
    argparser.add_argument('-f','--filterConfidence',dest='filterConfidence', metavar='FILTER', type=float,help='filter results with confidence below threshold',required=False,default=-1.0)
    argparser.add_argument('-s','--sfraction',dest='sfraction', metavar='SUPPORT_FILTER', type=float,help='sfraction filter for results ranked by confidence',required=False,default=-1.0)
    argparser.add_argument('--keepLeastConfident',dest='sfKeepLeastConf',action=argparse.BooleanOptionalAction,default=False,help='Instead of most confident, return least confident')
    argparser.add_argument('--perClassFiltering',dest='sfPerClass',action=argparse.BooleanOptionalAction,default=False,help='Filter sample-fraction from each class separately')
    argparser.add_argument('-r','--relabels',dest='relabels',type=str,required=False,help='Compute metrics with relabel file on classes',metavar='RELABEL',default=None)
    argparser.add_argument('--log_calibration',dest='log_calibration',type=str,required=False,help='Log confidence recalibration',metavar='LOGCAL',default=None)
    argparser.add_argument('--read_calibration',dest='read_calibration',type=str,required=False,help='Read confidence recalibration',metavar='READCAL',default=None)
    argparser.add_argument('--calFilterN', dest='calFilterN',metavar='CAL_FILTERN', type=int,help='Calibration filter length',required=False,default=40)
    argparser.add_argument('--calHistBins', dest='calHistBins',metavar='CAL_HISTBINS', type=int,help='Calibration histogram filter bins',required=False,default=-1)
    argparser.add_argument('-P','--perturb',dest='perturb', metavar='PERTURBANCE', type=float,help='Perturbation of truth for estimation of label error',required=False,default=0.0)
    

def perturbTruth(df,perturbance) :
    if perturbance > 0.0 :
       truths = df['Truth']
       psample = int(len(df)*perturbance)
       df_pert = df.sample(psample)
       df_pert['Truth'] = (df_pert['Truth']+1).mod(2)
       df_pert['Correct'] = (df_pert['Correct']+1).mod(2)
       df.loc[df_pert.index,:] = df_pert
    return df


def processParseResultsArguments(args) :
    # Ensure that parameters that can be specified multiple times, are condensed into simple list
    args.testn = [item for sublist in args.testn for item in sublist]
    args.splitn = [item for sublist in args.splitn for item in sublist]
    args.modeln = [item for sublist in args.modeln for item in sublist]


def getFile(args,prefix,testn,idx,mdx=-1) :
    # Start by assuming mdx arg is good
    file = f'{prefix}{testn}_{idx}_{mdx}.csv'
    #breakpoint()
    if not os.path.exists(file) :
        # If that doesn't work check first modeln[0] argument (if just one)
        if len(args.modeln) == 1 :
            file = f'{prefix}{testn}_{idx}_{args.modeln[0]}.csv'
        if not os.path.exists(file) :
            file = f'{prefix}{testn}_{idx}_0.csv'
        if not os.path.exists(file) :
            file = f'{prefix}{testn}_{idx}.csv'
        if not os.path.exists(file) :
            file = f'{prefix}{testn}.csv'
        if not os.path.exists(file) :
            return None
    return file


def expSeries(filterN) :
    coef = np.exp(np.linspace(0,1,filterN+1))-1
    coef = coef[1:]
    coef = coef / coef.sum()
    return coef


def estimateCalibConf(correctnessV,filterN) :
    # Create Exponential filter
    coef = expSeries(filterN)
    calconfvals = np.convolve(correctnessV,np.flip(coef))
    #calconfvals = calconfvals[filterN-1:] # start at filterN element
    calconfvals = calconfvals[0:-(filterN-1)] # start at filterN element
    # Series may not be strictly monotonically increasing (so sort it!)
    calconfvals = np.sort(calconfvals)
    # And, finally go ahead and normalize curve, so that we set highest
    # confidence to 1, and lowest to 0
    #calconfvals = calconfvals - calconfvals[0]
    #calconfvals = calconfvals / calconfvals[-1]
    return calconfvals


##    numIntermediateBuckets = int(numSamples - bucketSize*(numBins-1) - (bucketSize-1))
#def buckets(numSamples,numBins) :
#    bucketSize = int(np.floor((numSamples-1)/(numBins-1)))
#    numIntermediateBuckets = int(numSamples - bucketSize*(numBins-1) - (bucketSize))
#    if numIntermediateBuckets < 0 :
#        numIntermediateBuckets = 0
#    numNormalBuckets = numBins - numIntermediateBuckets - 1
#    intermediateBucketSize = bucketSize + 1
#    lastBucketSize = int(numSamples - bucketSize*numNormalBuckets - intermediateBucketSize*numIntermediateBuckets)
# 
#    return (bucketSize,numNormalBuckets,numIntermediateBuckets,lastBucketSize)



def buckets(numSamples,numBins) :
    assert numSamples >= numBins
    bucketSize = int(np.floor(numSamples/numBins))
    numExcess = numSamples - bucketSize*(numBins-1)
    numLongBuckets = 0
    if numExcess > bucketSize :
        numLongBuckets = numExcess - bucketSize
        assert numBins > numLongBuckets
        numNormalBuckets = numBins - numLongBuckets
    else :
        numNormalBuckets = numBins
    return (bucketSize,numNormalBuckets,numLongBuckets)

 

##    numIntermediateBuckets = int(numSamples - bucketSize*(numBins-1) - (bucketSize-1))
#def buckets(numSamples,numBins) :
#    assert numSamples >= numBins
#
#    bucketSize = int(np.1floor((numSamples)/(numBins)))
#    numExcess = numSamples - bucketSize*(numBins-1)
#    numLongBuckets = 0
#    numShortBuckets = 0
#    if numExcess == bucketSize :
#
#    if numExcess > bucketSize :
#        numLongBuckets = 1 + (numExcess - bucketSize)
#        assert numBins > numLongBuckets
#        numNormalBuckets = numBins - numLongBuckets
#        #if numExcess < (numBins-1) :
#        #    numLongBuckets = numExcess
#        #    numNormalBuckets = numBins-numLongBuckets
#        #elif numExcess < bucketSize :
#       
#    elif numExcess < bucketSize:
#        numShortBuckets = bucketSize - numExcess
#        assert numBins > numShortBuckets 
#        numNormalBuckets = numBins - numShortBuckets
#        #if numExcess < (numBins-1) :
#        #    numShortBuckets = numExcess
#    else :
#        # Then we are done and bucketSize perfectly divides
#        numNormalBuckets = numBins
#
#    return (bucketSize,numNormalBuckets,numShortBuckets,numLongBuckets)
#
#    #if bucketSize/2 < numExcess :
#
#    #else :
#
#    #numIntermediateBuckets = int(numSamples - bucketSize*(numBins-1) - (bucketSize))
#    #if numIntermediateBuckets < 0 :
#    #    numIntermediateBuckets = 0
#    #numNormalBuckets = numBins - numIntermediateBuckets - 1
#    #intermediateBucketSize = bucketSize + 1
#    #lastBucketSize = int(numSamples - bucketSize*numNormalBuckets - intermediateBucketSize*numIntermediateBuckets)
# 
#    #return (bucketSize,numNormalBuckets,numIntermediateBuckets,lastBucketSize)


def histogramCalibConf(dfn,numBins) :
    # We'll separate data buckets to in this case
    # as opposed to linspaced buckets
    bucketSize,numNormalBuckets,numLongBuckets = buckets(len(dfn),numBins)
    longBucketSize = bucketSize + 1
    #bucketSize = int(np.floor((len(dfn)-1)/(numBins-1)))
    ##numIntermediateBuckets = int(np.floor((len(dfn)-bucketSize*(numBins-1))/bucketSize - 1))
    #numIntermediateBuckets = int(len(dfn) - bucketSize*(numBins-1) - (bucketSize-1))
    #if numIntermediateBuckets < 0 :
    #    numIntermediateBuckets = 0
    #numNormalBuckets = numBins - numIntermediateBuckets - 1
    #intermediateBucketSize = bucketSize + 1
    #lastBucketSize = int(len(dfn) - bucketSize*numNormalBuckets - intermediateBucketSize*numIntermediateBuckets)
    #breakpoint()
    # Calculate actual accuracy for each bucket
    newConfidence = []
    #for i in range(0,numBins-1) :
    lastni = 0
    for i in range(0,numNormalBuckets) :
        startidx = int(i*bucketSize)
        actualConf = dfn['Correct'][startidx:startidx+bucketSize].mean()
        newConfidence.append(np.ones((bucketSize,1))*actualConf)
        lastni = i+1
    lastii = 0
    for i in range(0,numLongBuckets) :
        startidx = int(lastni*bucketSize + i*longBucketSize)
        actualConf = dfn['Correct'][startidx:startidx+longBucketSize].mean()
        newConfidence.append(np.ones((longBucketSize,1))*actualConf)
        lastii = i+1
    # And now do the last bin (which may have slightly fewer elements)
    #startidx = int(lastni*bucketSize + lastii*longBucketSize)
    #actualConf = dfn['Correct'][startidx:].mean()
    #newConfidence.append(np.ones((lastBucketSize,1))*actualConf)
    calconfvals = np.concatenate(newConfidence)
    return calconfvals


def precalibratedConf(calMap,dfn) :
    #breakpoint()
    lastIndex=len(calMap)-1
    calValues = np.array(calMap['Confidence'])
    uncalValues = np.array(calMap['UncalibratedConfidence'])
    updatedConf = []
    for conf in dfn['UncalibratedConfidence'] :
        idx = bisect.bisect_left(uncalValues,conf)
        idx = min(idx,lastIndex) # Need to handle special case, idx > lastIndex
        updatedConf.append(calValues[idx])
    return updatedConf


# Current method is essentially a histogram equalization
# Note, the true algorithm will need to find a mapping function (e.g. a polynomial or piecewise linear)
def calibrateConfidence(df,args,testn,splitn,modeln) :
    df = df.reset_index(drop=True)
    df['Correct'] = (df['Truth'] == df['Outcome']).astype(int)
    #breakpoint()
    #df.loc[df.last_valid_index()+1] = [df.last_valid_index()+1,'Synthetic0',1,0.5,0.0,0,0] # ensure we have a zero and ensure we have a 1 so that everything is normalized properly later.
    #df.loc[df.last_valid_index()+1] = [df.last_valid_index()+1,'Synthetic1',1,1.0,1.0,1,1]
    if len(df.columns) == 6:
        df.loc[df.last_valid_index()+1] = ['Synthetic0',1,0.5,0.0,0,0] # ensure we have a zero and ensure we have a 1 so that everything is normalized properly later.
        df.loc[df.last_valid_index()+1] = ['Synthetic1',1,1.0,1.0,1,1]
    elif len(df.columns) == 8:
        df.loc[df.last_valid_index()+1] = ['Synthetic0',1,True,0.5,df.loc[0]['RANSAC_Thresh'],0.0,0,0] # ensure we have a zero and ensure we have a 1 so that everything is normalized properly later.
        df.loc[df.last_valid_index()+1] = ['Synthetic0',1,True,1.0,df.loc[0]['RANSAC_Thresh'],1.0,1,1] # ensure we have a zero and ensure we have a 1 so that everything is normalized properly later.
    else :
        assert "DF column number unknown"
    confvals = df['Confidence'].sort_values(ascending=True)
    #calconfvals = np.linspace(0,1,len(confvals))
    df['UncalibratedConfidence'] = df['Confidence']
    dfn = df.loc[confvals.index]
    #breakpoint()
    if args.read_calibration is not None :
        calMap = readConfidenceMap(args,testn,splitn,modeln)
        dfn['Confidence'] = precalibratedConf(calMap,dfn)
    elif args.calHistBins > 0 :
        dfn['Confidence'] = histogramCalibConf(dfn,args.calHistBins)
    else :
        # TODO: IS THIS CORRECT? >>>>>>>>>>>>>vv dfn['Correct']
        dfn['Confidence'] = estimateCalibConf(dfn['Correct'],args.calFilterN)
    df = dfn.sort_index()
    return df


def constructedFilename(filename,testn,splitn,modeln) :
    if type(testn) is list :
        if len(testn) == 1 :
            testn = testn[0]
        else :
            testn = -1

    if type(splitn) is list :
        if len(splitn) == 1 :
            splitn = splitn[0]
        else :
            splitn = -1

    if type(modeln) is list :
        if len(modeln) == 1 :
            modeln = modeln[0]
        else :
            modeln = -1

    if testn > -1 :
        filename = f'{filename}_{testn}'
    if splitn > -1 :
        filename = f'{filename}_{splitn}'
    if modeln > -1 and os.path.exists(f'{filename}_{modeln}.csv') :
        filename = f'{filename}_{modeln}'
    filename = f'{filename}.csv'
    return filename


def saveConfidenceMap(df,args,testn,splitn,modeln) :
    if args.log_calibration is not None:
        filename = constructedFilename(args.log_calibration,testn,splitn,modeln)
        df = df[['Prediction','Confidence','Truth','Outcome']]
        df.to_csv(filename,index=False)


def readConfidenceMap(args,testn,splitn,modeln) :
    if args.read_calibration is not None:
        #breakpoint()
        filename = constructedFilename(args.read_calibration,testn,splitn,modeln)
        df = pd.read_csv(filename)
        df['UncalibratedConfidence'] = np.abs(df['Prediction']-0.5)*2
        confvals = df['UncalibratedConfidence'].sort_values(ascending=True)
        dfn = df.loc[confvals.index]
        # In this routine, we want to return in confidence order
        #   so we do not return to original index order
        # df = dfn.sort_index() # DON'T DO THIS!
        return dfn
    else :
        return None


def filterLowConfidence(df,filterConfidence) :
    currRows = len(df)
    df = df[df['Confidence'] > filterConfidence]
    #df = df[df['Confidence'] < filterConfidence] # If wanting to filter by confidence
    prunedRows = len(df)
    print(f'Filtered {currRows-prunedRows} of total {currRows} from data')
    return df


# Make perClass default to False
def filterBySampleFraction(df,sfraction,keepSamples=False,perClass=False,keepLeastConf=False) :
    
    if perClass :
        #breakpoint()
        dfneg = df[df['Truth'] == 0]
        dfpos = df[df['Truth'] == 1]
        confvalsneg = dfneg['Confidence'].sort_values(ascending=(not keepLeastConf)) # pass keepLeastConf=True to filter least confident performing samples
        confvalspos = dfpos['Confidence'].sort_values(ascending=(not keepLeastConf))
        dfnneg = dfneg.loc[confvalsneg.index]
        dfnpos = dfpos.loc[confvalspos.index]
        numToFilterneg = int(len(dfnneg) * (1.0 - sfraction))
        numToFilterpos = int(len(dfnpos) * (1.0 - sfraction))
        if keepSamples :
            dfnneg['FILTERED'] = False
            dfnneg.loc[dfnneg.iloc[0:numToFilterneg].index,'FILTERED'] = True
            dfnpos['FILTERED'] = False
            dfnpos.loc[dfnpos.iloc[0:numToFilterpos].index,'FILTERED'] = True
        else :
            dfnneg = dfnneg.iloc[numToFilterneg:]
            dfnpos = dfnpos.iloc[numToFilterpos:]
        dfn = pd.concat([dfnneg,dfnpos],axis=0).sort_index()
        df = dfn.sort_index() # but don't reindex in case we want result to be merged with other model
        print(f'Filtered {numToFilterneg} of total {len(confvalsneg)} from negative class')
        print(f'Filtered {numToFilterpos} of total {len(confvalspos)} from positive class')
    else :
        confvals = df['Confidence'].sort_values(ascending=(not keepLeastConf)) # pass keepLeastConf=True to filter least confident performing samples
        dfn = df.loc[confvals.index]
        #dfn = dfn.reset_index(drop=True)
        numToFilter = int(len(dfn) * (1.0 - sfraction))
        if keepSamples :
            dfn['FILTERED'] = False
            # Now set percentage to True
            dfn.loc[dfn.iloc[0:numToFilter].index,'FILTERED'] = True
        else :
            # Remove first numToFilter samples
            dfn = dfn.iloc[numToFilter:]
        df = dfn.sort_index() # but don't reindex in case we want result to be merged with other model
        print(f'Filtered {numToFilter} of total {len(confvals)} from data')
    return df


def filterSamples(df,args) :
    if args.sfraction > 0.0 :
        return filterBySampleFraction(df,args.sfraction,perClass=args.sfPerClass,keepLeastConf=args.sfKeepLeastConf)
    elif args.filterConfidence > 0.0 :
        return filterLowConfidence(df,args.filterConfidence)
    #elif args.filterByMetrics # Will this be possible? ... unlikely (instead will need search/seek algo)
    else :
        return df


def concatDataFrame(f,df,args,testn,splitn,modeln=-1) :
    if os.path.exists(f) :
        print(f'Adding {f}')
        res = pd.read_csv(f)
        res['Confidence']=np.abs(res['Prediction']-0.5)*2
        res['Truth'] = readLabels(res['PatchName'],relabels=args.relabels,stripPaths=True)
        #if args.read_calibration is not None or args.log_calibration is not None :
        #    res = calibrateConfidence(res,args,testn,splitn,modeln)
        df = pd.concat((df,res))
    else :
        print(f'WARNING: {f} does not exist')
    return df


def appendDataFrame(file,resList,args,testn,splitn,modeln) :
    if os.path.exists(file) :
        print(f'Adding {file}')
        res = pd.read_csv(file)
        if 'Unnamed: 0' in res.columns:
            res = res.drop(columns=['Unnamed: 0'])
        res['Confidence']=np.abs(res['Prediction']-0.5)*2
        res['Truth'] = readLabels(res['PatchName'],relabels=args.relabels,stripPaths=True)
        if args.read_calibration is not None or args.log_calibration is not None :
            res = calibrateConfidence(res,args,testn,splitn,modeln)
        resList.append(res)
        return res
    else :
        print(f'WARNING: {file} does not exist')
        return None

def reduceEnsemble(resList,args) :
    iterativeMerge = False
    if iterativeMerge :
        left = resList[0]
        i = 1
        for right in resList[1:] :
            left = pd.merge(left,right,on='PatchName',suffixes=[None,f'_{i}'])
            i += 1
        #left = left.rename({'Outcome':'Outcome_0','Prediction':'Prediction_0','Confidence':'Confidence_0'},axis=1)
        #left = left.rename({'Outcome_0':'Outcome','Prediction_0':'Prediction','Confidence_0':'Confidence'},axis=1)
        left = left.rename({'Outcome_1':'Outcome','Prediction_1':'Prediction','Confidence_1':'Confidence'},axis=1)
        left = left.rename({'Outcome_2':'Outcome','Prediction_2':'Prediction','Confidence_2':'Confidence'},axis=1)
        left = left.rename({'Outcome_3':'Outcome','Prediction_3':'Prediction','Confidence_3':'Confidence'},axis=1)
        left = left.rename({'Outcome_4':'Outcome','Prediction_4':'Prediction','Confidence_4':'Confidence'},axis=1)
        res = left
    else :
        res = pd.concat(resList,axis=1,join="inner")
        res = res.reset_index(drop=True)
    res["EnsembleOutcome"] = res["Outcome"].median(axis=1)
    res["SumOutcome"] = res["Outcome"].sum(axis=1)
    res["EnsemblePrediction"] = res["Prediction"].mean(axis=1)
    res["EnsembleConfidence"] = res["Confidence"].median(axis=1)
    numFiltered = 0
    if args.filterConfidence > 0 :
        for i, row in res.iterrows():
            outcomes = row["Outcome"].reset_index(drop=True)
            predictions = row["Prediction"].reset_index(drop=True)
            confidences = row["Confidence"].reset_index(drop=True)
            if args.smartEnsemble :
                strengths = row["Confidence"].reset_index(drop=True)
                # Important Note: these confidence values have not been calibrated
                # making this technique suspect.
                goodIndcs = strengths > args.filterConfidence
                goodIndcs = goodIndcs[goodIndcs].index
                if len(goodIndcs) > 0 :
                    outcomesf = outcomes[goodIndcs]
                    predictionsf = predictions[goodIndcs]
                    poutcome = res.at[i,"EnsembleOutcome"]
                    pprediction = res.at[i,"EnsemblePrediction"]
                    res.at[i,"EnsembleOutcome"] = outcomesf.median()
                    res.at[i,"SumOutcome"] = outcomesf.sum()
                    #res.at[i,"EnsemblePrediction"] = predictionsf.mean()
                    res.at[i,"EnsemblePrediction"] = predictionsf.median()
                    if len(goodIndcs) < 5 :
                        print(f'Row {i} filtered indices: {[idx for idx in goodIndcs]}, {[o for o in outcomesf]}, {poutcome},{res.at[i,"EnsembleOutcome"]},{[p for p in predictions]},{[s for s in strengths]}')
                else :
                    numFiltered += 1
                    res.at[i,"EnsembleOutcome"] = outcomes.median()
                    res.at[i,"SumOutcome"] = outcomes.sum()
                    res.at[i,"EnsemblePrediction"] = predictions.median() # This used to be mean (but that was probably a bad choice as one bad outlier could greatly effect center)
                    print(f'Row {i} filtered ALL indices; using all columns')
            else :
                res.at[i,"EnsembleOutcome"] = outcomes.median()
                res.at[i,"SumOutcome"] = outcomes.sum()
                res.at[i,"EnsemblePrediction"] = predictions.median()
                res.at[i,"EnsembleConfidence"] = confidences.median()
                
    # Default is voting scheme, but could use a prediction averaging scheme too (but doesn't seem to work as well)
    res["EnsembleOutcome2"] = res["EnsemblePrediction"] > 0.5
    df = pd.DataFrame()
    df["PatchName"] = res["PatchName"].iloc[:,0] 
    df["Outcome"] = res["EnsembleOutcome"]
    df["OutcomeType2"] = res["EnsembleOutcome2"]
    df["Prediction"] = res["EnsemblePrediction"]
    df["RANSAC_Thresh"] = res["SumOutcome"] # may be used to detect ensemble performance
    df['Confidence']=res["EnsembleConfidence"]
    df['Truth'] = readLabels(df['PatchName'],relabels=args.relabels,stripPaths=True)

    return df


def fixDataFrame(df) :
    df = df.reset_index(drop=True)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.reset_index(drop=True)
    return df


# Generator that yields DataFrame (df) and Test# (testn) on each iteration
# On each iteration, metrics may be measured with: 
#   acc,f1,auc=computeBinomialMetrics(df['Outcome'],df['Prediction'],df['Truth'],numModels=1,modelNumber=testn)
def parseResults(prefix,args) :
    
    if args.model_1st_ensemble :
        for mdx in args.modeln :
            for testn in args.testn :
                print(f'Computing For Holdout Test: {testn}')
                
                if args.ensemble :
                    resList = []
                    for idx in args.splitn :
                        fi = getFile(args,prefix,testn,idx,mdx)
                        if fi is not None:
                            appendDataFrame(fi,resList,args,testn,idx,mdx)
                    
                    df = reduceEnsemble(resList,args)
                else :
                    assert False, 'FATAL ERROR: model_1st_ensemble requires --ensemble arg'

                # This block of code is suspect, shouldn't be in model_1st_ensemble without
                # actually doing an ensemble I believe!
                #else :
                #    df = pd.DataFrame()
                #    for idx in args.splitn :
                #        df = concatDataFrame(getFile(args,prefix,testn,idx,mdx),df)
                #    
                #    df = fixDataFrame(df)
                
                #df['Truth'] = readLabels(df['PatchName'],relabels=args.relabels,stripPaths=True)
                df = df.dropna()
                df = fixDataFrame(df)
                if args.read_calibration is None or args.log_calibration is not None :
                    df = calibrateConfidence(df,args,testn,-1,mdx)
                #df = filterLowConfidence(df,args.filterConfidence)
                df = filterSamples(df,args)
                saveConfidenceMap(df,args,testn,-1,mdx)

                # If asked to, randomly perturb the solution by flopping Truth and Correct(ness)
                df = perturbTruth(df,args.perturb)

                #yield (df,testn,mdx,None) # this looks like a latent bug, modded to below
                yield (df,testn,args.splitn,mdx)
                
    elif args.split_1st_ensemble :
        for idx in args.splitn :
            for testn in args.testn :
                print(f'Computing For Holdout Test: {testn}')
                
                if args.ensemble :
                    resList = []
                    if len(args.modeln) > 1 :
                        for mdx in args.modeln :
                            fi = getFile(args,prefix,testn,idx,mdx)
                            if fi is not None:
                                appendDataFrame(fi,resList,args,testn,idx,mdx)
                    else:
                        assert False, 'FATAL ERROR: have to have multiple models to support ensemble'
                    df = reduceEnsemble(resList,args)
                
                else :
                    assert False, 'FATAL ERROR: model_1st_ensemble requires --ensemble arg'
                # This block of code is suspect, shouldn't be in model_1st_ensemble without
                # actually doing an ensemble I believe!
                #else :
                #    df = pd.DataFrame()
                #    if len(args.modeln) > 1 :
                #        for mdx in args.modeln :
                #            df = concatDataFrame(getFile(args,prefix,testn,idx,mdx),df)
                #    else:
                #        df = concatDataFrame(getFile(args,prefix,testn,idx),df)
                #    
                #    df = fixDataFrame(df)
                
                #df['Truth'] = readLabels(df['PatchName'],relabels=args.relabels,stripPaths=True)
                df = df.dropna()
                df = fixDataFrame(df)
                if args.read_calibration is None or args.log_calibration is not None :
                    df = calibrateConfidence(df,args,testn,idx,args.modeln)
                #df = filterLowConfidence(df,args.filterConfidence)
                df = filterSamples(df,args)
                saveConfidenceMap(df,args,testn,idx,args.modeln)

                # If asked to, randomly perturb the solution by flopping Truth and Correct(ness)
                df = perturbTruth(df,args.perturb)

                yield (df,testn,idx,args.modeln)
    
    elif args.ransacScore :
        resList = []
        ensembleSupport = 0
        #if len(args.modeln) > 0 :
        for testn in args.testn :
            for mdx in args.modeln :
                for idx in args.splitn :
                    fi = getFile(args,prefix,testn,idx,mdx)
                    if fi is not None:
                        appendDataFrame(fi,resList,args,testn,idx,mdx)
                        ensembleSupport += 1

        df = reduceEnsemble(resList,args)
        
        #df['Truth'] = readLabels(df['PatchName'],relabels=args.relabels,stripPaths=True)
        df.loc[df['Truth']==False,'RANSAC_Thresh'] = ensembleSupport - df[df['Truth'] == False]['RANSAC_Thresh']
        df['RANSAC_Thresh'] /= ensembleSupport
        print(f'Ensemble Support: {ensembleSupport}')
 
        df = df.dropna()
        df = fixDataFrame(df)
        
        # If asked to, randomly perturb the solution by flopping Truth and Correct(ness)
        df = perturbTruth(df,args.perturb)

        # Only returning once here but doing this so that it works in for loop
        yield (df,'All',args.splitn,args.modeln)

    else :
        for testn in args.testn :
            print(f'Computing For Holdout Test: {testn}')
            
            if args.ensemble :
                resList = []
                ensembleSupport = 0
                if len(args.modeln) > 0 :
                    for mdx in args.modeln :
                        for idx in args.splitn :
                            fi = getFile(args,prefix,testn,idx,mdx)
                            if fi is not None:
                                appendDataFrame(fi,resList,args,testn,idx,mdx)
                                ensembleSupport += 1
                else :
                    for idx in args.splitn :
                        fi = getFile(args,prefix,testn,idx)
                        if fi is not None:
                            appendDataFrame(fi,resList,args,testn,idx)
                            ensembleSupport += 1

                df = reduceEnsemble(resList,args)
                
                #df['Truth'] = readLabels(df['PatchName'],relabels=args.relabels,stripPaths=True)
                df.loc[df['Truth']==False,'RANSAC_Thresh'] = ensembleSupport - df[df['Truth'] == False]['RANSAC_Thresh']
                df['RANSAC_Thresh'] /= ensembleSupport
                print(f'Ensemble Support: {ensembleSupport}')
            
            else :
                #breakpoint()
                df = pd.DataFrame()
                if len(args.modeln) > 0 :
                    for mdx in args.modeln :
                        for idx in args.splitn :
                            fi = getFile(args,prefix,testn,idx,mdx)
                            if fi is not None:
                                df = concatDataFrame(fi,df,args,testn,idx,mdx)
                else:
                    for idx in args.splitn :
                        fi = getFile(args,prefix,testn,idx)
                        if fi is not None:
                            df = concatDataFrame(fi,df,args,testn,idx)
                
                df = fixDataFrame(df)
                
                #df['Truth'] = readLabels(df['PatchName'],relabels=args.relabels,stripPaths=True)
            
            
            df = df.dropna()
            df = fixDataFrame(df)
            if args.read_calibration is None or args.log_calibration is not None :
                #breakpoint()
                df = calibrateConfidence(df,args,testn,args.splitn,args.modeln)
            #breakpoint()
            df = filterSamples(df,args)
            #df = filterLowConfidence(df,args.filterConfidence)
            saveConfidenceMap(df,args,testn,args.splitn,args.modeln)

            # If asked to, randomly perturb the solution by flopping Truth and Correct(ness)
            df = perturbTruth(df,args.perturb)

            yield (df,testn,args.splitn,args.modeln)

