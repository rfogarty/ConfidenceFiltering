
from stats.MetricTypes import ConfMatrix,Performance
from stats.PrintMetrics import printBounds
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score,f1_score,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve,RocCurveDisplay
import random
import pandas as pd
import matplotlib.pyplot as pl
 

def computeStats(cm) :
    TP=cm.TP
    TN=cm.TN
    FP=cm.FP
    FN=cm.FN
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    sensitivity=TP/(TP+FN)
    specificity=TN/(TN+FP)
    PPV=TP/(TP+FP)
    precision=PPV
    NPV=TN/(TN+FN)
    F1=2*TP/(2*TP+FP+FN)
    return Performance(cm,accuracy,sensitivity,specificity,PPV,NPV,F1)

 #Sensitivity TP/(TP+FN)	Specificity TN/(TN+FP)	Precision - TP/(TP+FP)	F1 (Dice) 2*P*S/(P+S)
def derivedStats(dataframe,prefix='') :
   TP=dataframe.true_positives
   TN=dataframe.true_negatives
   FP=dataframe.false_positives
   FN=dataframe.false_negatives
   sensitivity=TP/(TP+FN)
   specificity=TN/(TN+FP)
   auc=(specificity+sensitivity)/2
   #print(f'sensitivity.shape ({sensitivity.dtype}):{sensitivity.shape}')
   #print(f'specificity.shape ({specificity.dtype}):{specificity.shape}')
   #print(f'auc.shape ({auc.dtype}):{auc.shape}')
   precision=TP/(TP+FP)
   dice=precision*sensitivity/(precision+sensitivity)
   dice=dice.mul(2.0)
   kappa=((TP*TN)-(FN*FP))/(((TP+FP)*(FP+TN))+((TP+FN)*(FN+TN)))
   kappa=kappa.mul(2.0)
   altStats=pd.DataFrame({f'{prefix}sensitivity':sensitivity,f'{prefix}specificity':specificity,
                          f'{prefix}alt_auc':auc,f'{prefix}precision':precision,f'{prefix}dice':dice,f'{prefix}kappa':kappa})
   return altStats

def derivedTrainStats(dataframe) :
   return derivedStats(dataframe)

def derivedValStats(dataframe) :
   return derivedStats(dataframe,prefix='val_')

   
def f1stat(cm) :
    F1=2*cm.TP/(2*cm.TP+cm.FP+cm.FN)
    return F1

# This function was developed to compute the upper and lower bounds
# or confidence limits as described in Data Mining Section 5.3
def computeBounds(acc,numgood,numbad,z=0.69) :
    N = numgood + numbad
    f = numbad/N
    z2 = z*z
    N2 = N*N
    f2 = f*f
    lower = (f + (z2/(2.0*N)) + (z * m.sqrt((f/N) - (f2/N) + (z2/(4.0*N2)))))/(1.0 + (z2/N))
    upper = (f + (z2/(2.0*N)) - (z * m.sqrt((f/N) - (f2/N) + (z2/(4.0*N2)))))/(1.0 + (z2/N))
    print(f"Total accuracy bounds : ({acc-lower}:{acc+upper})")
    return (upper,lower)


def confusionMatrixToMetrics(confMat) :
    T = np.sum(confMat)
    P = np.sum(np.diag(confMat))
    N = T - P
    # First compute accuracy
    accuracy = P/T
    # Next calculate recall (aka sensitivity)
    rank=confMat.shape[0]
    recall=[0]*rank
    averageRecall=0
    for i in range(rank) :
        recall[i]=confMat[i,i]/np.sum(confMat[i,:])
        averageRecall+=recall[i]
    averageRecall=averageRecall/rank
    # Next calculate precision (aka positive predictive value)
    precision=[0]*rank
    averagePrecision=0
    for i in range(rank) :
        precision[i]=confMat[i,i]/np.sum(confMat[:,i])
        averagePrecision+=precision[i]
    averagePrecision=averagePrecision/rank
    # Next calculate F1
    f1score=2*averagePrecision*averageRecall/(averagePrecision+averageRecall)

    print(f'Confusion Matrix   :\n{confMat}\n')
    print('Metrics per class')
    for i in range(rank) :
        print(f'    Recall[{i}]    : {recall[i]}')
        print(f'    Precision[{i}] : {precision[i]}')
     
    print(f'\nAverage Recall   : {averageRecall}')
    print(f'Average Precision  : {averagePrecision}')
    print(f'Accuracy           : {accuracy}')
    print(f'F1-score           : {f1score}\n')


def computeBinomialMetrics(outcomes,predictions,test_labels,numModels=10,modelNumber=0,confidences=None) :
    #predicted = model.predict(test_data)
    #print(f'test_labels.shape: {test_labels.shape}')
    #test_labels = np.argmax(test_labels, axis=1)
    #breakpoint()
    
    # before doing anything let's stable sort everything
    # as the below requires that the classes are sorted for the
    # bootstrap logic to work properly.
    outcomes = np.array(outcomes)
    predictions = np.array(predictions)
    test_labels = np.array(test_labels)
    retConfidence = True
    if confidences is None:
        retConfidence = False
        confidences = np.ones((len(outcomes),1))
    sorted_indices=np.argsort(test_labels,kind='stable')
    outcomes = outcomes[sorted_indices]
    predictions = predictions[sorted_indices]
    test_labels = test_labels[sorted_indices]
    confidences = confidences[sorted_indices]

    #breakpoint()
    minQuorum=int((numModels+2)/2)
    predicted = (outcomes >= minQuorum).astype(int)
    #if args.multiTask :
    #    #print(f'predicted[0].shape: {predicted[0].shape}')
    #    top2accuracy = top_k_accuracy_score(test_labels, predicted[0],k=2)
    #    top3accuracy = top_k_accuracy_score(test_labels, predicted[0],k=3)
    #    predicted = np.argmax(predicted[0], axis=1)
    #else :
    #    #print(f'predicted.shape: {predicted.shape}')
    #    top2accuracy = top_k_accuracy_score(test_labels, predicted,k=2)
    #    top3accuracy = top_k_accuracy_score(test_labels, predicted,k=3)
    #    predicted = np.argmax(predicted, axis=1)

    print(f'predictions.shape:{predictions.shape}')
    print(f'test_labels.shape:{test_labels.shape}')
    #predictions = np.reshape(predictions,(predictions.shape[1],predictions.shape[2]))
    print(f'predictions.shape:{predictions.shape}')

    numNegative = np.sum(outcomes == 0)
    numPositive = np.sum(outcomes != 0)
    assert (numNegative + numPositive) == outcomes.shape[0], 'numNegative+numPositive != outcomes'
    #numEachLabel = int(predictions.shape[0]/2)
    #print(f'numEachLabel={numEachLabel}')
    print(f'numNegative,numPositive={numNegative},{numPositive}')
    
    c1=[x for x in range(numNegative)]
    c2=[x for x in range(numNegative,numNegative+numPositive)]
    sampleSize=int(0.75*min(numNegative,numPositive))
    numBootstraps=100
    accuracyList=[]
    precisionC1List=[]
    recallC1List=[]
    f1scoreC1List=[]
    precisionC2List=[]
    recallC2List=[]
    f1scoreC2List=[]
    precisionMacroList=[]
    recallMacroList=[]
    f1scoreMacroList=[]
    precisionMicroList=[]
    recallMicroList=[]
    f1scoreMicroList=[]
    precisionBinaryList=[]
    recallBinaryList=[]
    f1scoreBinaryList=[]
    aucList=[]
    lowestConfidences=[]
    i = 0
    # Note: some bootstraps may be numerically unstable
    # So we will ignore those.
    imax = 0
    while i < numBootstraps :
        random.shuffle(c1)
        random.shuffle(c2)
        sample=c1[:sampleSize] + c2[:sampleSize]
        predictedSample = predicted[sample]
        labelSample = test_labels[sample]
        predictionsSample = predictions[sample]
        consample=confidences[sample]
        try :
            lowestConfidence = np.min(consample)
            accuracy = accuracy_score(labelSample,predictedSample)
            prfs = precision_recall_fscore_support(labelSample,predictedSample)
            prfsBin = precision_recall_fscore_support(labelSample,predictedSample,average='binary')
            rocAUC = roc_auc_score(labelSample,predictionsSample)
            lowestConfidences.append(lowestConfidence)
            accuracyList.append(accuracy)
            #precisionC1List.append(prfs[0][0])
            #precisionC2List.append(prfs[0][1])
            #recallC1List.append(prfs[1][0])
            #recallC2List.append(prfs[1][1])
            #f1scoreC1List.append(prfs[2][0])
            #f1scoreC2List.append(prfs[2][1])
            #prfs = precision_recall_fscore_support(labelSample,predictedSample,average='macro')
            #precisionMacroList.append(prfs[0])
            #recallMacroList.append(prfs[1])
            #f1scoreMacroList.append(prfs[2])
            #prfs = precision_recall_fscore_support(labelSample,predictedSample,average='micro')
            #precisionMicroList.append(prfs[0])
            #recallMicroList.append(prfs[1])
            #f1scoreMicroList.append(prfs[2])
            #precisionBinaryList.append(prfsBin[0])
            #recallBinaryList.append(prfsBin[1])
            f1scoreBinaryList.append(prfsBin[2])
            aucList.append(rocAUC)
            i = i + 1
        except :
            print('Incomplete sample')
            imax = imax + 1
            if imax > 10*numBootstraps :
                break

    # Computed for all (don't know if I do anything with this!)
    #precisionMacro = precision_score(test_labels, predicted, average='macro')
    #precisionMicro = precision_score(test_labels, predicted, average='micro')
    #precisionBinary = precision_score(test_labels, predicted, average='binary')
    #recallMacro = recall_score(test_labels, predicted, average='macro')
    #recallMicro = recall_score(test_labels, predicted, average='micro')
    #recallBinary = recall_score(test_labels, predicted, average='binary')
    #f1Macro = f1_score(test_labels, predicted, average='macro')
    #f1Micro = f1_score(test_labels, predicted, average='micro')
    #f1Binary = f1_score(test_labels, predicted, average='binary')
    #accuracy = accuracy_score(test_labels, predicted)
    #confMat = confusion_matrix(test_labels, predicted)
    #prfs = precision_recall_fscore_support(test_labels,predicted)

    if retConfidence:
        return (accuracyList,f1scoreBinaryList,aucList,lowestConfidences)
    else :
        return (accuracyList,f1scoreBinaryList,aucList)
#    roc_fpr,roc_tpr,roc_thresholds = roc_curve(test_labels,predictions,drop_intermediate=False)
#
#    # Save off data in Pandas DataFrame to CSV
#    pdROC=pd.DataFrame({'thresholds':roc_thresholds,'roc_fpr':roc_fpr,'roc_tpr':roc_tpr})
#    pdROC.to_csv(f'ROC_Data{modelNumber}.csv',index=False) 
#
#    pl.clf()
#    rocgraph = RocCurveDisplay(fpr=roc_fpr,tpr=roc_tpr)
#    rocgraph.plot()
#    pl.savefig(f'ROC_Model{modelNumber}.png')
#
#    auc = roc_auc_score(test_labels,predictions)
#    print("\n                       Performance Metrics")
#    print("---------------------------------------------------------------------")
#    printBounds('Accuracy',accuracyList)
#    printBounds('F1',f1scoreBinaryList)
#    print(f'Accuracy         : {accuracy}')
#    #print(f'Accuracy (top-2) : {top2accuracy}')
#    #print(f'Accuracy (top-3) : {top3accuracy}')
#    print(f'Recall           : {recallMacro}(macro), {recallMicro}(micro), {recallBinary}(binary)')
#    print(f'Precision        : {precisionMacro}(macro), {precisionMicro}(micro), {precisionBinary}(binary)')
#    print(f'F1 score         : {f1Macro}(macro), {f1Micro}(micro),{f1Binary}(binary)')
#    print(f'PRFS             : {prfs}')
#    print(f'AUC              : {auc}')
#    #print(f'ROC.fpr          : {roc_fpr}')
#    #print(f'ROC.tpr          : {roc_tpr}')
#    #print(f'ROC.thresholds   : {roc_thresholds}')
#    #print(f'ROC_fpr.shape    : {roc_fpr.shape}')
#    #print(f'ROC_tpr.shape    : {roc_tpr.shape}')
#    #print(f'ROC_thresholds.shape    : {roc_thresholds.shape}')
#    print(f'Confusion Matrix : \n{confMat}')
#    print(f'BootstrappedAccuracy    : \n{accuracyList}')
#    print(f'BootstrappedPrecisionC1 : \n{precisionC1List}')
#    print(f'BootstrappedPrecisionC2 : \n{precisionC2List}')
#    print(f'BootstrappedRecallC1    : \n{recallC1List}')
#    print(f'BootstrappedRecallC2    : \n{recallC2List}')
#    print(f'BootstrappedF1scoreC1   : \n{f1scoreC1List}')
#    print(f'BootstrappedF1scoreC2   : \n{f1scoreC2List}')
#    print(f'BootstrappedPrecisionMacro : \n{precisionMacroList}')
#    print(f'BootstrappedRecallMacro    : \n{recallMacroList}')
#    print(f'BootstrappedF1scoreMacro   : \n{f1scoreMacroList}')
#    print(f'BootstrappedPrecisionMicro : \n{precisionMicroList}')
#    print(f'BootstrappedRecallMicro    : \n{recallMicroList}')
#    print(f'BootstrappedF1scoreMicro   : \n{f1scoreMicroList}')
#    print(f'BootstrappedPrecisionBinary : \n{precisionBinaryList}')
#    print(f'BootstrappedRecallBinary    : \n{recallBinaryList}')
#    print(f'BootstrappedF1scoreBinary   : \n{f1scoreBinaryList}')
#    print(f'BootstrappedAUC         : \n{aucList}')
#    print(f'BootstrappedMeanAccuracy        : \n{np.mean(accuracyList)}')
#    print(f'BootstrappedMeanPrecisionC1     : \n{np.mean(precisionC1List)}')
#    print(f'BootstrappedMeanPrecisionC2     : \n{np.mean(precisionC2List)}')
#    print(f'BootstrappedMeanRecallC1        : \n{np.mean(recallC1List)}')
#    print(f'BootstrappedMeanRecallC2        : \n{np.mean(recallC2List)}')
#    print(f'BootstrappedMeanF1scoreC1       : \n{np.mean(f1scoreC1List)}')
#    print(f'BootstrappedMeanF1scoreC2       : \n{np.mean(f1scoreC2List)}')
#    print(f'BootstrappedMeanPrecisionMacro  : \n{np.mean(precisionMacroList)}')
#    print(f'BootstrappedMeanRecallMacro     : \n{np.mean(recallMacroList)}')
#    print(f'BootstrappedMeanF1scoreMacro    : \n{np.mean(f1scoreMacroList)}')
#    print(f'BootstrappedMeanPrecisionMicro  : \n{np.mean(precisionMicroList)}')
#    print(f'BootstrappedMeanRecallMicro     : \n{np.mean(recallMicroList)}')
#    print(f'BootstrappedMeanF1scoreMicro    : \n{np.mean(f1scoreMicroList)}')
#    print(f'BootstrappedMeanPrecisionBinary : \n{np.mean(precisionBinaryList)}')
#    print(f'BootstrappedMeanRecallBinary    : \n{np.mean(recallBinaryList)}')
#    print(f'BootstrappedMeanF1scoreBinary   : \n{np.mean(f1scoreBinaryList)}')
#    print(f'BootstrappedMeanAUC             : \n{np.mean(aucList)}')


