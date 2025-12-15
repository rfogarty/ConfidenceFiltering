import argparse
from configurable.dataParameters import *


def addCommonArguments(parser,BS) :
    img_rows, img_cols, img_channels = tensorSize()
    tileRank=3
    numPools=1
    # Image control
    prescale = 1.0

    parser.add_argument('-g','--gpus',dest='gpus',type=int,nargs='+',required=False,default=(0),
                        help='GPUs to use for accelerating program',metavar='GPUS')
    parser.add_argument('-b','--bs',dest='bs',type=int,required=False,default=BS,
                        help='Batch size when training',metavar='BATCH')
    parser.add_argument('-P','--numPools',dest='numPools',type=int,required=False,default=numPools,
                        help='Number of Pools to split the data each epoch',metavar='POOLS')
    parser.add_argument('-T','--test',dest='test',type=int,required=True,help='test for this training',metavar='TEST')

    parser.add_argument('-t','--tensor-shape',dest='tensor_shape',type=int,nargs=3,required=False,
                        help='tensor shape for input images',metavar='TENSOR_SHAPE',default=[img_rows,img_cols,img_channels])
    parser.add_argument('-r','--rank',dest='rank',type=int,required=False,
                        help='rank of cutmix tile matrix for input images',metavar='RANK',default=tileRank)
    parser.add_argument('--mask',action=argparse.BooleanOptionalAction,default=False,
                        help='Whether to enable masks on input images')
    parser.add_argument('--gray',action=argparse.BooleanOptionalAction,default=False,
                        help='Whether to force images to grayscale')
    #parser.add_argument('--mask', action='store_true',
    #                    help='Whether to enable masks on input images')
    #parser.add_argument('--no-mask', action='store_false',
    #                    help='Whether to disable masks on input images')
    #parser.set_defaults(mask=False)
    parser.add_argument('--prescale',dest='prescale',type=float,required=False,
                        default=prescale,help='prescale images by factor',metavar='PRESCALE')


def addTrainingArguments(parser,trainFeatureLayers=False) :

    augmentData=True
    learningRate=0.05
    lrReductionRate=0
    maxNumEpochs=500
    snapshotCycle=60
    batchSize=30
    earlyStopPatience=-1
    parser.add_argument('-s','--split',dest='split',type=int,required=True,help='split for this training',metavar='SPLIT')
    parser.add_argument('--augment',action=argparse.BooleanOptionalAction,default=augmentData,
                        help='Whether to enable augmentation during training')
    parser.add_argument('-e','--epochs',dest='epochs',type=int,required=False,default=maxNumEpochs,
                        help='Maximum number of epochs to train',metavar='EPOCHS')
    parser.add_argument('-c','--cycle',dest='cycle',type=int,required=False,default=snapshotCycle,
                        help='Cycle time for cosine-annealing when training',metavar='CYCLE')
    parser.add_argument('-l','--lr',dest='lr',type=float,required=False,default=learningRate,
                        help='Learning rate used for training',metavar='LR')
    parser.add_argument('--lrRedRate',dest='lrRedRate',type=float,required=False,default=lrReductionRate,
                        help='Learning rate reduction rate',metavar='LR_REDUCTION_RATE')
    parser.add_argument('-p','--patience',dest='patience',type=int,required=False,default=earlyStopPatience,
                        help='Early stop patience when training',metavar='PATIENCE')
    parser.add_argument('--trainFeatures',action=argparse.BooleanOptionalAction,default=trainFeatureLayers,
                        help='Whether to enable training feature layers')


def processCommandLine() :

    batchSize=30
    parser = argparse.ArgumentParser(description='Train Neural Net for some split.')
    addTrainingArguments(parser)
    addCommonArguments(parser,batchSize)
    args=parser.parse_args()
    if args.patience == -1 :
        args.patience=int(args.cycle*1.25)
    print('Training on Split: ' + str(args.split))
    print(f'Configuration Arguments:\n{args}\n')
    
    return args


def processCommandLineFineTune() :

    batchSize=30
    trainFeatureLayers=True
    parser = argparse.ArgumentParser(description='Train Neural Net for some split.')
    addTrainingArguments(parser,trainFeatureLayers=True)
    addCommonArguments(parser,batchSize)
    parser.add_argument('-R','--restart',dest='restart',type=int,required=True,
                        help='Epoch to restart from',metavar='RESTART_EPOCH')
    parser.add_argument('modelfiles', metavar='MODEL', type=str, nargs='+',help='a model (weights) file for the deep network')

    args=parser.parse_args()
    if args.patience == -1 :
        args.patience=int(args.cycle*1.25)
    print('Training on Split: ' + str(args.split))
    print(f'Configuration Arguments:\n{args}\n')
    
    return args


def processCommandLineEnsembleTest() :
    batchSize=5
    parser = argparse.ArgumentParser(description='Test Neural Net for splits.')
    addCommonArguments(parser,batchSize)
    parser.add_argument('modelfiles', metavar='MODEL', type=str, nargs='+',help='a model (weights) file for the deep network')
    parser.add_argument('--chooseR',dest='chooseR', metavar='N_CHOOSE_R',type=int,required=False,help='Compute ensemble stats for all combinations R of N',default=0)
    args=parser.parse_args()
    print('Testing for models: ' + args.modelfiles[0] + '...')
    print(f'Configuration Arguments:\n{args}\n')
    
    return args


def processCommandLineTest() :
    batchSize=5
    parser = argparse.ArgumentParser(description='Test Neural Net for splits.')
    addCommonArguments(parser,batchSize)
    parser.add_argument('modelfiles', metavar='MODEL', type=str, nargs='+',help='a model (weights) file for the deep network')
    args=parser.parse_args()
    print('Testing for models: ' + args.modelfiles[0] + '...')
    print(f'Configuration Arguments:\n{args}\n')
    
    return args


def processCommandLineSplitTest() :
    batchSize=5
    parser = argparse.ArgumentParser(description='Test Neural Net for split.')
    parser.add_argument('-s','--split',dest='split',type=int,required=True,help='split for this test',metavar='SPLIT')
    parser.add_argument('-n','--number',dest='number',type=int,required=False,default=-1,help='snapshot model number for this test',metavar='NUMBER')
    addCommonArguments(parser,batchSize)
    parser.add_argument('modelfiles', metavar='MODEL', type=str, nargs='+',help='a model (weights) file for the deep network')
    args=parser.parse_args()
    print('Testing for models: ' + args.modelfiles[0] + '...')
    print(f'Configuration Arguments:\n{args}\n')
    
    return args

def processCommandLineExtractFeats() :
    batchSize=5
    parser = argparse.ArgumentParser(description='Test Neural Net for split.')
    parser.add_argument('-s','--split',dest='split',type=int,required=True,help='split for this test',metavar='SPLIT')
    parser.add_argument('-n','--number',dest='number',type=int,required=False,default=-1,help='snapshot model number for this test',metavar='NUMBER')
    addCommonArguments(parser,batchSize)
    parser.add_argument('modelfiles', metavar='MODEL', type=str, nargs='+',help='a model (weights) file for the deep network')
    parser.add_argument('--distilledFeats',action=argparse.BooleanOptionalAction,default=False,
                        help='Whether to distill the features (from first FCN layer)')
    args=parser.parse_args()
    print('Testing for models: ' + args.modelfiles[0] + '...')
    print(f'Configuration Arguments:\n{args}\n')
    
    return args


def processCommandLineTestData() :
    batchSize=5
    parser = argparse.ArgumentParser(description='Test Data Iterators')
    parser.add_argument('-s','--split',dest='split',type=int,required=True,help='split for this training',metavar='SPLIT')
    parser.add_argument('-c','--composite',dest='comp',type=str,required=False,help='composite image processing technique',metavar='COMP',default='default')
    addCommonArguments(parser,batchSize)
    #parser.add_argument('modelfiles', metavar='MODEL', type=str, nargs='+',help='a model (weights) file for the deep network')
    args=parser.parse_args()
    #print('Testing for models: ' + args.modelfiles[0] + '...')
    print(f'Configuration Arguments:\n{args}\n')
    
    return args


