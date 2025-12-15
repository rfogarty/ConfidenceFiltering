import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append(os.getcwd())
nthreads = "32"
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ['TF_NUM_INTEROP_THREADS'] = nthreads
os.environ['TF_NUM_INTRAOP_THREADS'] = nthreads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math as m
from configurable.dataPresentation import *
from configurable.dataParameters import *
from data import *
from model import *
from gpuassign import *
from arguments import *
from stats import *
import pandas as pd

def testModel(args) :

    imageProcessing,tensorShape = getImageProcessing(args)
    print(f'Configured IP mode: {str(imageProcessing)}')
    dataIterator,labels,paths = getAllTraining(imageProcessing,args,tensorShape)

    sampleStats=dict.fromkeys(paths,0)

    vggStages = 5;
    shrinkStages = vggStages - dlNetworkStages()
 
    for modelfile in args.modelfiles :
        if numClasses() == 2 :
            outcomes,predictions = makeBinomialInferences(dataIterator,modelfile,args.tensor_shape,numClasses(),args,shrinkStages=shrinkStages)
            #breakpoint()
            results = pd.DataFrame(list(zip(paths,np.squeeze(outcomes),np.squeeze(predictions))),columns=['PatchName','Outcome','Prediction'])
            if args.number >= 0 :
                results.to_csv(f'Results_{args.test}_{args.split}_{args.number}.csv')
            else :
                results.to_csv(f'Results_{args.test}_{args.split}.csv')
            #correct,incorrect = computeEnsembleMetrics(outcomes,labels,paths,sampleStats,1)
            #accuracy,totcorrect,totincorrect = printAccuracies(correct,incorrect)
            #computeBounds(accuracy,totcorrect,totincorrect)
            #computeBinomialMetrics(outcomes,predictions,labels,numModels=1,modelNumber=args.split)
        elif numClasses() > 2 :
            outcomes,predictions = makeMultinomialInferences(dataIterator,modelfile,args.tensor_shape,numClasses(),args,shrinkStages=shrinkStages)
            outcomes_b = relabel(outcomes)
            labels_b = relabel(labels)
            predictions_b = binarizePredictions(predictions)
            results = pd.DataFrame(list(zip(paths,outcomes_b,predictions_b)),columns=['PatchName','Outcome','Prediction'])
            #results = pd.DataFrame(list(zip(paths,outcomes,predictions)),columns=['PatchName','Outcome','Prediction'])
            if args.number >= 0 :
                results.to_csv(f'Results_{args.test}_{args.split}_{args.number}.csv')
            else :
                results.to_csv(f'Results_{args.test}_{args.split}.csv')
            #correct,incorrect = computeEnsembleMetrics(outcomes_b,labels_b,paths,sampleStats,1)
            #accuracy,totcorrect,totincorrect = printAccuracies(correct,incorrect)
            #computeBounds(accuracy,totcorrect,totincorrect)
            #computeBinomialMetrics(outcomes_b,predictions_b,labels_b,numModels=1,modelNumber=args.split)
            #computeMultinomialMetrics(predictions,labels)
            #correct,incorrect = computeMultinomialEnsembleMetrics(predictions,numClasses(),labels,paths,sampleStats,1)
        else :
            print(f'numClasses({numClasses()} must by >= 2')

    #with open(f'InferenceForHoldoutNoRANSAC{args.split}.txt', 'w') as f:
    #    for path,correct in sampleStats.items() :
    #        print(f'{path}: {float(correct)/len(args.modelfiles)}', file=f)
 
############################################################################################
# Code to compute accuracy on a Test Set - this should only be done once training and test
# on a validation set shows promise.

if __name__ == "__main__" :

    # Process arguments from commandline
    args = processCommandLineSplitTest()

    # Limit which GPUs we can run on
    assignGPUs(args.gpus)

    # Train our model
    testModel(args)
