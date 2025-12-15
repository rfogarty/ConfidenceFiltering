
import numpy as np
import pandas as pd
import math as m
import random
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score,f1_score,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve,RocCurveDisplay

from configurable.dataParameters import *


# Since the ensemble was not set up as Keras layer, metrics were computed by hand.
# This simple routine compares the outcomes to the expected labels and computes
# the number correct and incorrect for each individual class.
def computeEnsembleMetrics(outcomes,test_labels,paths,sampleStats,numModels) :
    correct=[0]*2
    incorrect=[0]*2
    minQuorum = int((numModels+1)/2) # If there are multiple models voting, there must be a min majority
    # Next compare predicted outcomes with the labels and add to appropriate
    # correct/incorrect table index.
    for pred,label,path in zip(outcomes,test_labels,paths):
        labelval = int(label)
        if labelval == 0 :
            numCorrect = numModels - pred
        else :
            numCorrect = pred
        if (numCorrect >= minQuorum) : correct[labelval] += 1
        else : incorrect[labelval] += 1
        sampleStats[path] += numCorrect
    
    return (correct,incorrect)


def computeBinomialMetrics(outcomes,predictions,test_labels,numModels=10,modelNumber=0) :
    #predicted = model.predict(test_data)
    #print(f'test_labels.shape: {test_labels.shape}')
    #test_labels = np.argmax(test_labels, axis=1)
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
    predictions = np.reshape(predictions,(predictions.shape[1],predictions.shape[2]))
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
    for i in range(numBootstraps) :
        random.shuffle(c1)
        random.shuffle(c2)
        sample=c1[:sampleSize] + c2[:sampleSize]
        predictedSample = predicted[sample]
        labelSample = test_labels[sample]
        predictionsSample = predictions[sample]
        accuracyList.append(accuracy_score(labelSample,predictedSample))
        prfs = precision_recall_fscore_support(labelSample,predictedSample)
        precisionC1List.append(prfs[0][0])
        precisionC2List.append(prfs[0][1])
        recallC1List.append(prfs[1][0])
        recallC2List.append(prfs[1][1])
        f1scoreC1List.append(prfs[2][0])
        f1scoreC2List.append(prfs[2][1])
        prfs = precision_recall_fscore_support(labelSample,predictedSample,average='macro')
        precisionMacroList.append(prfs[0])
        recallMacroList.append(prfs[1])
        f1scoreMacroList.append(prfs[2])
        prfs = precision_recall_fscore_support(labelSample,predictedSample,average='micro')
        precisionMicroList.append(prfs[0])
        recallMicroList.append(prfs[1])
        f1scoreMicroList.append(prfs[2])
        prfs = precision_recall_fscore_support(labelSample,predictedSample,average='binary')
        precisionBinaryList.append(prfs[0])
        recallBinaryList.append(prfs[1])
        f1scoreBinaryList.append(prfs[2])
        aucList.append(roc_auc_score(labelSample,predictionsSample))

    precisionMacro = precision_score(test_labels, predicted, average='macro')
    precisionMicro = precision_score(test_labels, predicted, average='micro')
    precisionBinary = precision_score(test_labels, predicted, average='binary')
    recallMacro = recall_score(test_labels, predicted, average='macro')
    recallMicro = recall_score(test_labels, predicted, average='micro')
    recallBinary = recall_score(test_labels, predicted, average='binary')
    f1Macro = f1_score(test_labels, predicted, average='macro')
    f1Micro = f1_score(test_labels, predicted, average='micro')
    f1Binary = f1_score(test_labels, predicted, average='binary')
    accuracy = accuracy_score(test_labels, predicted)
    confMat = confusion_matrix(test_labels, predicted)
    prfs = precision_recall_fscore_support(test_labels,predicted)
    roc_fpr,roc_tpr,roc_thresholds = roc_curve(test_labels,predictions,drop_intermediate=False)

    # Save off data in Pandas DataFrame to CSV
    pdROC=pd.DataFrame({'thresholds':roc_thresholds,'roc_fpr':roc_fpr,'roc_tpr':roc_tpr})
    pdROC.to_csv(f'ROC_Data{modelNumber}.csv',index=False) 

    pl.clf()
    rocgraph = RocCurveDisplay(fpr=roc_fpr,tpr=roc_tpr)
    rocgraph.plot()
    pl.savefig(f'ROC_Model{modelNumber}.png')

    auc = roc_auc_score(test_labels,predictions)
    print("\n                       Performance Metrics")
    print("---------------------------------------------------------------------")
    print(f'Accuracy         : {accuracy}')
    #print(f'Accuracy (top-2) : {top2accuracy}')
    #print(f'Accuracy (top-3) : {top3accuracy}')
    print(f'Recall           : {recallMacro}(macro), {recallMicro}(micro), {recallBinary}(binary)')
    print(f'Precision        : {precisionMacro}(macro), {precisionMicro}(micro), {precisionBinary}(binary)')
    print(f'F1 score         : {f1Macro}(macro), {f1Micro}(micro),{f1Binary}(binary)')
    print(f'PRFS             : {prfs}')
    print(f'AUC              : {auc}')
    #print(f'ROC.fpr          : {roc_fpr}')
    #print(f'ROC.tpr          : {roc_tpr}')
    #print(f'ROC.thresholds   : {roc_thresholds}')
    #print(f'ROC_fpr.shape    : {roc_fpr.shape}')
    #print(f'ROC_tpr.shape    : {roc_tpr.shape}')
    #print(f'ROC_thresholds.shape    : {roc_thresholds.shape}')
    print(f'Confusion Matrix : \n{confMat}')
    print(f'BootstrappedAccuracy    : \n{accuracyList}')
    print(f'BootstrappedPrecisionC1 : \n{precisionC1List}')
    print(f'BootstrappedPrecisionC2 : \n{precisionC2List}')
    print(f'BootstrappedRecallC1    : \n{recallC1List}')
    print(f'BootstrappedRecallC2    : \n{recallC2List}')
    print(f'BootstrappedF1scoreC1   : \n{f1scoreC1List}')
    print(f'BootstrappedF1scoreC2   : \n{f1scoreC2List}')
    print(f'BootstrappedPrecisionMacro : \n{precisionMacroList}')
    print(f'BootstrappedRecallMacro    : \n{recallMacroList}')
    print(f'BootstrappedF1scoreMacro   : \n{f1scoreMacroList}')
    print(f'BootstrappedPrecisionMicro : \n{precisionMicroList}')
    print(f'BootstrappedRecallMicro    : \n{recallMicroList}')
    print(f'BootstrappedF1scoreMicro   : \n{f1scoreMicroList}')
    print(f'BootstrappedPrecisionBinary : \n{precisionBinaryList}')
    print(f'BootstrappedRecallBinary    : \n{recallBinaryList}')
    print(f'BootstrappedF1scoreBinary   : \n{f1scoreBinaryList}')
    print(f'BootstrappedAUC         : \n{aucList}')
    print(f'BootstrappedMeanAccuracy        : \n{np.mean(accuracyList)}')
    print(f'BootstrappedMeanPrecisionC1     : \n{np.mean(precisionC1List)}')
    print(f'BootstrappedMeanPrecisionC2     : \n{np.mean(precisionC2List)}')
    print(f'BootstrappedMeanRecallC1        : \n{np.mean(recallC1List)}')
    print(f'BootstrappedMeanRecallC2        : \n{np.mean(recallC2List)}')
    print(f'BootstrappedMeanF1scoreC1       : \n{np.mean(f1scoreC1List)}')
    print(f'BootstrappedMeanF1scoreC2       : \n{np.mean(f1scoreC2List)}')
    print(f'BootstrappedMeanPrecisionMacro  : \n{np.mean(precisionMacroList)}')
    print(f'BootstrappedMeanRecallMacro     : \n{np.mean(recallMacroList)}')
    print(f'BootstrappedMeanF1scoreMacro    : \n{np.mean(f1scoreMacroList)}')
    print(f'BootstrappedMeanPrecisionMicro  : \n{np.mean(precisionMicroList)}')
    print(f'BootstrappedMeanRecallMicro     : \n{np.mean(recallMicroList)}')
    print(f'BootstrappedMeanF1scoreMicro    : \n{np.mean(f1scoreMicroList)}')
    print(f'BootstrappedMeanPrecisionBinary : \n{np.mean(precisionBinaryList)}')
    print(f'BootstrappedMeanRecallBinary    : \n{np.mean(recallBinaryList)}')
    print(f'BootstrappedMeanF1scoreBinary   : \n{np.mean(f1scoreBinaryList)}')
    print(f'BootstrappedMeanAUC             : \n{np.mean(aucList)}')



def computeMultinomialMetrics(predictions,test_labels) :
#def testModel(args,model,test_data,test_labels) :
    #predicted = model.predict(test_data)
    #print(f'test_labels.shape: {test_labels.shape}')
    test_labels = np.argmax(test_labels, axis=1)
    outcomes = np.argmax(predictions,axis=1)
    #if args.multiTask :
    #    #print(f'outcomes[0].shape: {outcomes[0].shape}')
    #    top2accuracy = top_k_accuracy_score(test_labels, outcomes[0],k=2)
    #    top3accuracy = top_k_accuracy_score(test_labels, outcomes[0],k=3)
    #    outcomes = np.argmax(outcomes[0], axis=1)
    #else :
    #    #print(f'outcomes.shape: {outcomes.shape}')
    #    top2accuracy = top_k_accuracy_score(test_labels, outcomes,k=2)
    #    top3accuracy = top_k_accuracy_score(test_labels, outcomes,k=3)
    #    outcomes = np.argmax(outcomes, axis=1)

    # The following maybe NOOPs or they may aggregate labels into smaller set of labels.
    test_labels = relabel(test_labels)
    outcomes = relabel(outcomes)

    precisionMacro = precision_score(test_labels, outcomes, average='macro')
    precisionMicro = precision_score(test_labels, outcomes, average='micro')
    recallMacro = recall_score(test_labels, outcomes, average='macro')
    recallMicro = recall_score(test_labels, outcomes, average='micro')
    f1Macro = f1_score(test_labels, outcomes, average='macro')
    f1Micro = f1_score(test_labels, outcomes, average='micro')
    accuracy = accuracy_score(test_labels, outcomes)
    # Ensure that labels are in the appropriate order
    #usedOutcomes=list(set(np.concatenate((outcomes,test_labels))))
    #usedLabels=classLabels()[usedOutcomes]
    #lb = LabelBinarizer()
    #lb.fit(usedLabels)
    #breakpoint()
    confMat = confusion_matrix(test_labels, outcomes)
    print("\n                       Performance Metrics")
    print("---------------------------------------------------------------------")
    print(f'Accuracy         : {accuracy}')
    #print(f'Accuracy (top-2) : {top2accuracy}')
    #print(f'Accuracy (top-3) : {top3accuracy}')
    print(f'Recall           : {recallMacro}(macro), {recallMicro}(micro)')
    print(f'Precision        : {precisionMacro}(macro), {precisionMicro}(micro)')
    print(f'F1 score         : {f1Macro}(macro), {f1Micro}(micro)')
    print(f'Confusion Matrix : \n{confMat}')



def computeBinomialEnsembleMetrics(outcomes,test_labels,paths,sampleStats,numModels) :
    correct=[0]*2
    incorrect=[0]*2
    #test_labels = np.argmax(test_labels, axis=1)
    minQuorum=int((numModels+2)/2)
    outcomes = (outcomes >= minQuorum).astype(int)
    # Next compare predicted outcomes with the labels and add to appropriate
    # correct/incorrect table index.
    for pred,label,path in zip(outcomes,test_labels,paths):
        labelval = int(label)
        if labelval == pred :
            correct[labelval] += 1
            sampleStats[path] += 1
        else : incorrect[labelval] += 1
   
    # Compute consensus score confusion matrix
    currentSubject='UNINIT'
    consensusConfusion=np.zeros((2,2))
    subjectPredictionMap={}
    #subjectPrediction=[0]*4
    #subjectLabel=-1
    for pred,label,path in zip(outcomes,test_labels,paths):
        subject = path2subject(path)
        labelval = int(label)
        if subject in subjectPredictionMap :
            subjectLabel,subjectPrediction=subjectPredictionMap[subject]
            if subjectLabel != labelval :
                print(f'WARNING: subjectLabel:{subjectLabel} has changed {labelval}')
        else :
            subjectPredictionMap[subject]=(labelval,[0]*2)
            _,subjectPrediction=subjectPredictionMap[subject]
        #print(f'type(pred)={type(pred)}, pred={pred}')
        subjectPrediction[pred[0]]+=1
    
    for subject,(subjectLabel,subjectPrediction) in subjectPredictionMap.items() :
        predict=np.argmax(subjectPrediction)
        print(f'Subject[{subject}:{subjectLabel}]: {subjectPrediction}')
        consensusConfusion[subjectLabel,predict]+=1
    
    print('Consensus Metrics')
    print('-----------------')
    confusionMatrixToMetrics(consensusConfusion)

    return (correct,incorrect)
 
 
def computeMultinomialEnsembleMetrics(predictions,numClasses,test_labels,paths,sampleStats,numModels,outcomes=None) :
    # Note: numClasses and numModels should be able to be inferred from the shape of the predictions array
    correct=[0]*numClasses
    incorrect=[0]*numClasses
    if test_labels.ndim > 1 :
        test_labels = np.argmax(test_labels, axis=1)
    if outcomes is None :
        outcomes=np.argmax(predictions, axis=1)
    # Next compare predicted outcomes with the labels and add to appropriate
    # correct/incorrect table index.
    for pred,label,path in zip(outcomes,test_labels,paths):
        labelval = int(label)
        if labelval == pred :
            correct[labelval] += 1
            sampleStats[path] += 1
        else : incorrect[labelval] += 1
   
    ## Compute consensus score confusion matrix
    #currentSubject='UNINIT'
    #consensusConfusion=np.zeros((numClasses,numClasses))
    #subjectPredictionMap={}
    ##subjectPrediction=[0]*numClasses
    ##subjectLabel=-1
    #for pred,label,path in zip(outcomes,test_labels,paths):
    #    subject = path2subject(path)
    #    labelval = int(label)
    #    if subject in subjectPredictionMap :
    #        subjectLabel,subjectPrediction=subjectPredictionMap[subject]
    #        if subjectLabel != labelval :
    #            print(f'WARNING: subjectLabel:{subjectLabel} has changed {labelval}')
    #    else :
    #        subjectPredictionMap[subject]=(labelval,[0]*numClasses)
    #        _,subjectPrediction=subjectPredictionMap[subject]
    #    subjectPrediction[pred]+=1
    #
    #for subject,(subjectLabel,subjectPrediction) in subjectPredictionMap.items() :
    #    predict=np.argmax(subjectPrediction)
    #    print(f'Subject[{subject}:{subjectLabel}]: {subjectPrediction}')
    #    consensusConfusion[subjectLabel,predict]+=1
    #
    #print('Consensus Metrics')
    #print('-----------------')
    #confusionMatrixToMetrics(consensusConfusion)

    return (correct,incorrect)


if __name__ == "__main__" :
    logsAll,logs=loadLogs()

    print(f'SampledAUC         : \n{aucList}')

