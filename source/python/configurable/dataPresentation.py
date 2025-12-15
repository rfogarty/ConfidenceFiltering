from data import *
from configurable.dataParameters import *


def dataManager() :
    ###########################################################
    ###########################################################
    ###########################################################
    # DATA ITERATOR FUNCTION TO OVERRIDE #
    #return getDataResizedIteratorFromListing
    return getDataCropIteratorFromListing
    #return getDataSamplemixIteratorFromListing
    #return getDataFramedIteratorFromListing
    ###########################################################
    ###########################################################
    ###########################################################
    ###########################################################


def getImageProcessing(args) :
    imageProcessing = ImageProcessing(prescale=args.prescale,grayed=True,normalized=True,composite=None)
    # Override tensor shape to account for extra channels
    return (imageProcessing,args.tensor_shape)


def initialWeights(args) :

    if hasattr(args,"modelfiles") :
        return getattr(args,"modelfiles")[0]
    else :
        # Nominally for transfer learning: return "imagenet"
        return None


def dlNetworkStages() :
    return 3


def getTrainingAndValidation(imageProcessing,args,tensorShape) :
    mask_dir = None
    mask_ext = None
    if args.mask :
        mask_dir=maskDir()
        mask_ext=maskExt()

    dm = dataManager()
    dataIterator,labels,paths = dm(dataDir(),trainingSet(args.test,args.split),
                                   blocklist=trainingBlocklists(args.test,args.split),
                                   shuffled=True,batch_size=args.bs,numPools=args.numPools,
                                   input_tensor_shape=tensorShape,imageProcessing=imageProcessing,
                                   mask_dir=mask_dir,mask_ext=mask_ext,name='TrainingDataIterator')

    vdataIterator,vlabels,vpaths = dm(dataDir(),validationSet(args.test,args.split),
                                      blocklist=validationBlocklists(args.test,args.split),
                                      shuffled=True,batch_size=args.bs,numPools=1,autoBalance=True,
                                      input_tensor_shape=tensorShape,imageProcessing=imageProcessing,
                                      mask_dir=mask_dir,mask_ext=mask_ext,name='ValidationDataIterator')

    return (dataIterator,labels,paths,vdataIterator,vlabels,vpaths)


def getTrainingAndValidationWithFlips(imageProcessing,args,tensorShape) :
    mask_dir = None
    mask_ext = None
    if args.mask :
        mask_dir=maskDir()
        mask_ext=maskExt()

    dm = dataManager()
    dataIterator,labels,paths = dm(dataDir(),trainingSet(args.test,args.split),
                                   blocklist=trainingBlocklists(args.test,args.split),relabels=trainingFliplists(args.test,args.split),
                                   shuffled=True,batch_size=args.bs,numPools=args.numPools,
                                   input_tensor_shape=tensorShape,imageProcessing=imageProcessing,
                                   mask_dir=mask_dir,mask_ext=mask_ext,name='TrainingDataIterator')

    vdataIterator,vlabels,vpaths = dm(dataDir(),validationSet(args.test,args.split),
                                      blocklist=validationBlocklists(args.test,args.split),relabels=validationFliplists(args.test,args.split),
                                      shuffled=True,batch_size=args.bs,numPools=1,autoBalance=True,
                                      input_tensor_shape=tensorShape,imageProcessing=imageProcessing,
                                      mask_dir=mask_dir,mask_ext=mask_ext,name='ValidationDataIterator')

    return (dataIterator,labels,paths,vdataIterator,vlabels,vpaths)


def getValidationWithFlips(imageProcessing,args,tensorShape) :
    mask_dir = None
    mask_ext = None
    if args.mask :
        mask_dir=maskDir()
        mask_ext=maskExt()

    dm = dataManager()
    vdataIterator,vlabels,vpaths = dm(dataDir(),validationSet(args.test,args.split),
                                      blocklist=validationBlocklists(args.test,args.split),relabels=validationFliplists(args.test,args.split),
                                      shuffled=True,batch_size=args.bs,numPools=1,autoBalance=True,
                                      input_tensor_shape=tensorShape,imageProcessing=imageProcessing,
                                      mask_dir=mask_dir,mask_ext=mask_ext,name='ValidationDataIterator')

    return (vdataIterator,vlabels,vpaths)


def getAllTraining(imageProcessing,args,tensorShape) :
    mask_dir = None
    mask_ext = None
    if args.mask :
        mask_dir=maskDir()
        mask_ext=maskExt()

    dm = dataManager()
    dataIterator,labels,paths = dm(dataDir(),trainingSet(args.test,args.split),
                                   blocklist=None,
                                   shuffled=False,batch_size=args.bs,numPools=1,autoBalance=False,
                                   input_tensor_shape=tensorShape,imageProcessing=imageProcessing,
                                   mask_dir=mask_dir,mask_ext=mask_ext,name='TrainingDataIterator')

    return (dataIterator,labels,paths)


def getAllTheData(imageProcessing,args,tensorShape) :
    mask_dir = None
    mask_ext = None
    if args.mask :
        mask_dir=maskDir()
        mask_ext=maskExt()

    dm = dataManager()
    dataIterator,labels,paths = dm(dataDir(),allTheData(args.test,args.split),
                                   blocklist=None,
                                   shuffled=False,batch_size=args.bs,numPools=1,autoBalance=False,
                                   input_tensor_shape=tensorShape,imageProcessing=imageProcessing,
                                   mask_dir=mask_dir,mask_ext=mask_ext,name='AllDataIterator')

    return (dataIterator,labels,paths)


def getHoldout(imageProcessing,args,tensorShape) :
    mask_dir = None
    mask_ext = None
    if args.mask :
        mask_dir=maskDir()
        mask_ext=maskExt()

    dm = dataManager()
    dataIterator,labels,paths = dm(dataDir(),validationSetAlt1(args.test,args.split),
                                   blocklist=validationBlocklistsAlt1(args.split),
                                   shuffled=False,batch_size=args.bs,numPools=1,autoBalance=False,
                                   input_tensor_shape=tensorShape,imageProcessing=imageProcessing,
                                   mask_dir=mask_dir,mask_ext=mask_ext,name='HoldoutDataIterator')
                                   #blocklist=validationBlocklistsAlt1(args.split),relabels=validationFliplists(args.test,args.split),

    return (dataIterator,labels,paths)
