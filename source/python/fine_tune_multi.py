# -*- coding: utf-8 -*-
"""
Created on Thur Dec 23 15:11:00 2021

@author: rfogarty
"""
import os
import sys
sys.path.append(os.getcwd())
nthreads = "32"
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ['TF_NUM_INTEROP_THREADS'] = nthreads
os.environ['TF_NUM_INTRAOP_THREADS'] = nthreads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from gpuassign import *
#assignNumCPUs(int(nthreads))

import math
from keras.callbacks import CSVLogger,ModelCheckpoint,EarlyStopping

from configurable.dataPresentation import *
from configurable.dataParameters import *
from model import *
from snapshot import *
from archive import *
from gpuassign import *
from arguments import *
from checkpoint import *
from stats.MetricPlots import *
import pandas as pd


def configCallbacks(args,paths) :
    #1 Configure Snapshot Ensemble (and Learning Rate Cosine Annealer)
    iterationsPerCycle = math.ceil((len(paths)/args.numPools)/args.bs) * args.cycle
    print(f'Iterations per Cycle: {iterationsPerCycle}')
    filepath = f"snapshot-weights-test{args.test}-split{args.split}" + "-{snapshot_epoch:03d}.hdf5"
    checkpoint = SnapshotEnsemble(filepath,iterationsPerCycle,args.lr,args.lrRedRate,verbose=True)
    #2 Configure Metrics Logger
    csv_logger = CSVLogger(f'log-test{args.test}-split{args.split}.csv', append=True, separator=',')
    #3 Configure ModelCheckpoints
    #bestfilepath = f"best-accuracy-test{args.test}-split{args.split}" + "-{epoch:03d}.hdf5"
    #cpMonitor,cpMode = checkpointMonitor1()
    #saveBest1=ModelCheckpoint(filepath=bestfilepath,monitor=cpMonitor,mode=cpMode,save_best_only=True)
    #bestfilepath = f"best-loss-test{args.test}-split{args.split}" + "-{epoch:03d}.hdf5"
    #cpMonitor,cpMode = checkpointMonitor2()
    #saveBest2=ModelCheckpoint(filepath=bestfilepath,monitor=cpMonitor,mode=cpMode,save_best_only=True)
    bestfilepath = f"best-comboloss-test{args.test}-split{args.split}" + "-{epoch:03d}.hdf5"
    monFunc = LossPlusValidationLoss()
    saveBest3=ProgrammableModelCheckpoint(filepath=bestfilepath,monitorFunc=monFunc,verbose=True)
    #4 Lastly, configure EarlyStopping
    #esMonitor,esMode=earlyStopMonitor()
    #earlyStop=EarlyStopping(monitor=esMonitor,mode=esMode,min_delta=0.0005,patience=args.patience)
    
    #callbacks_list = [checkpoint,csv_logger,saveBest1,saveBest2,saveBest3,earlyStop]
    #callbacks_list = [checkpoint,csv_logger,saveBest3,earlyStop]
    callbacks_list = [checkpoint,csv_logger,saveBest3]
    return callbacks_list



#def configCallbacks(args,paths) :
#    #1 Configure Snapshot Ensemble (and Learning Rate Cosine Annealer)
#    iterationsPerCycle = math.ceil((len(paths)/args.numPools)/args.bs) * args.cycle
#    print(f'Iterations per Cycle: {iterationsPerCycle}')
#    filepath = f"snapshot-weights-test{args.test}-split{args.split}" + "-{snapshot_epoch:03d}-{epoch:03d}.hdf5"
#    checkpoint = SnapshotEnsemble(filepath,iterationsPerCycle,args.lr,args.lrRedRate,verbose=True)
#    #2 Configure Metrics Logger
#    csv_logger = CSVLogger(f'log-test{args.test}-split{args.split}.csv', append=True, separator=',')
#    #3 Configure ModelCheckpoints
#    #bestfilepath = f"best-accuracy-test{args.test}-split{args.split}" + "-{epoch:03d}.hdf5"
#    #cpMonitor,cpMode = checkpointMonitor1()
#    #saveBest1=ModelCheckpoint(filepath=bestfilepath,monitor=cpMonitor,mode=cpMode,save_best_only=True)
#    #bestfilepath = f"best-loss-test{args.test}-split{args.split}" + "-{epoch:03d}.hdf5"
#    #cpMonitor,cpMode = checkpointMonitor2()
#    #saveBest2=ModelCheckpoint(filepath=bestfilepath,monitor=cpMonitor,mode=cpMode,save_best_only=True)
#    bestfilepath = f"best-comboloss-test{args.test}-split{args.split}" + "-{epoch:03d}.hdf5"
#    monFunc = LossPlusValidationLoss()
#    saveBest3=ProgrammableModelCheckpoint(filepath=bestfilepath,monitorFunc=monFunc,verbose=True)
#    #4 Lastly, configure EarlyStopping
#    #esMonitor,esMode=earlyStopMonitor()
#    #earlyStop=EarlyStopping(monitor=esMonitor,mode=esMode,min_delta=0.0005,patience=args.patience)
#    
#    #callbacks_list = [checkpoint,csv_logger,saveBest1,saveBest2,earlyStop]
#    callbacks_list = [checkpoint,csv_logger,saveBest3]
#    return callbacks_list

def trainModel(args) :

    imageProcessing,tensorShape = getImageProcessing(args)
    print(f'Configured IP mode: {str(imageProcessing)}')
    dataIterator,labels,paths,vdataIterator,vlabels,vpaths = getTrainingAndValidationWithFlips(imageProcessing,args,tensorShape)

    callbacks_list = configCallbacks(args,paths)
    
    vggStages = 5;
    shrinkStages = vggStages - dlNetworkStages()
 
    model = buildFineTuningNet(initialWeights(args),args.tensor_shape,numClasses(),gpuIds2Names(args.gpus),augment=args.augment,trainFeatureLayers=args.trainFeatures,shrinkStages=shrinkStages)
    H = model.fit(dataIterator,batch_size=args.bs,
                  epochs=args.epochs,
                  initial_epoch=args.restart,
                  validation_data=vdataIterator,
                  callbacks=callbacks_list, verbose=1)
    
    saveHistory(finetuneHistory(args.split),H)
    
    #plotMetric(f"accuracy-split{args.split}",'binary_accuracy',H)
    df = pd.DataFrame({'LearningRate':callbacks_list[0].lrates})
    plotMetricAndAux(f"accuracy-test{args.test}-split{args.split}",'binary_accuracy',H,df,'LearningRate')
    plotMetricAndAux(f"val_accuracy-test{args.test}-split{args.split}",'val_binary_accuracy',H,df,'LearningRate')


if __name__ == "__main__" :
    # Process arguments from commandline
    args = processCommandLineFineTune()

    # Limit which GPUs we can run on
    assignGPUs(args.gpus)

    # Train our model
    trainModel(args)

