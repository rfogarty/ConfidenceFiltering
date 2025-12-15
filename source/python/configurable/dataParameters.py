import os
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def dataDir() :
    #return '/data/CombinedKaggleUMMCC/RegionsED'
    #return '/raid/rfogarty/CombinedKaggleUMMCC/GS3vsGS4UMandMCC5x5Fold'
    return '/raid/rfogarty/CombinedKaggleUMMCC/GS3vsGS4UMandMCC5x5Fold/Relabeled'
    #return '/mnt/d/Data/CombinedKaggleUMMCC'

def maskDir() :
    return '/raid/rfogarty/CombinedKaggleUMMCC/GS3vsGS4UMandMCC5x5Fold/Masks'

def maskExt() :
    return '.png'

def tensorSize() :
    img_rows, img_cols, img_channels = 300,300,3
    return (img_rows,img_cols,img_channels)

def checkpointMonitor1() :
    monitor='val_binary_accuracy'
    mode='max'
    return (monitor,mode)

def checkpointMonitor2() :
    monitor='val_loss'
    mode='min'
    return (monitor,mode)

def earlyStopMonitor() :
    monitor='val_loss'
    mode='min'
    return (monitor,mode)

##################################################################
# Data sets
def trainingSet(test,split) :
    return f'moffittTest{test}TrainingValidation.list'
    #return 'trainingSetGS3andGS4.list'

def allTheData(test,split) :
    return f'moffittTraining.list'

def validationSet(test,split) :
    #return ['validationSplit24.list','validationSplit146.list',f'validationSplit{split}.list']
    #return ['validationSplit24.list','validationSplit146.list']
    #return f'validationSet{split}.list'
    return f'moffittTest{test}Fold{split}.list'

def validationSetAlt1(test,split) :
    return [f'moffittTest{test}Holdout.list']
    #return f'validation_set{split}_ummcc.list'

def validationSetAlt2(split) :
    return f'validation_set{split}_panda.list'

def validationSetAlt3(split) :
    return f'gs3_and_gs4.list'

def tuningSet(test,split) :
    return trainingSet(test,split)

# To be used for tests like RANSACing
def seenTestSet(split) :
    return trainingSet(split)

def holdoutTestSet(split) :
    return None

##################################################################
# Blocklists
def trainingBlocklists(test,split) :
    data_dir=dataDir()
    # Currently, this is set up to block nothing.
    #blocklists=[f'{data_dir}/validationSplit24.list',f'{data_dir}/validationSplit146.list',f'{data_dir}/validationSplit{split}.list','blocklist.txt']
    #blocklists=[f'{data_dir}/moffittBlocklist.txt',f'blocklistTest{test}.list',f'{data_dir}/moffittTest{test}Fold{split}.list']
    blocklists=[f'{data_dir}/moffittBlocklist.txt',f'blocklist.list',f'{data_dir}/moffittTest{test}Fold{split}.list']
    # Here are blocklists when performing GS3 vs GS4 tests
    #blocklists=[f'{data_dir}/Kaggle/gs0-5_blocklist.txt',f'{data_dir}/validation_set{split}.list']
    return blocklists

def trainingFliplists(test,split) :
    data_dir=dataDir()
    #fliplists=[f'fliplistTest{test}.list']
    fliplists=[f'fliplist.list']
    return fliplists

def validationBlocklists(test,split) :
    data_dir=dataDir()
    blocklists=None
    #blocklists=[f'{data_dir}/moffittBlocklist.txt',f'blocklistTest{test}.list']
    blocklists=[f'{data_dir}/moffittBlocklist.txt',f'blocklist.list']
    #blocklists=[f'{data_dir}/Kaggle/gs0-5_blocklist.txt']
    return blocklists

def validationBlocklistsAlt1(split) :
    data_dir=dataDir()
    #blocklists=None
    blocklists=[f'{data_dir}/moffittBlocklist.txt']
    return blocklists

def validationBlocklistsAlt2(split) :
    return validationBlocklists(split)

def validationBlocklistsAlt3(split) :
    data_dir=dataDir()
    blocklists=None
    #blocklists=[f'{data_dir}/blocklistTooSmall.txt',f'{data_dir}/blocklistTooLarge.txt']
    return blocklists

def validationFliplists(test,split) :
    data_dir=dataDir()
    # Currently, this is set up to block nothing.
    #fliplists=[f'fliplistTest{test}.list']
    fliplists=[f'fliplist.list']
    return fliplists

def tuningBlocklists(test,split) :
    blocklists=trainingBlocklists(test,split)
    #blocklists=None
    #blocklists.append('blocklist.txt')
    return blocklists

def seenTestBlocklists(split) :
    return validationBlocklists(split)

def holdoutBlocklists(split) :
    return None

##################################################################
#  History files
def trainingHistory(split) :
    return f'trainHistoryDict-split{split}.pickle'

def finetuneHistory(split) :
    return f'finetuneHistoryDict-split{split}.pickle'

##################################################################
#  Path to Label converters
def numClasses() :
    #return 6
    #return 4
    return 2

def classLabels() :
    #classes = ['GS3-PANDA','GS3-UMMCC','GS3-MOFFITT','GS4-PANDA','GS4-UMMCC','GS4-MOFFITT']
    #classes = ['GS3-UMMCC','GS3-MOFFITT','GS4-UMMCC','GS4-MOFFITT']
    classes = ['GS3-MOFFITT','GS4-MOFFITT']
    # Process through LabelBinarizer to ensure consistency with reporting tools
    lb = LabelBinarizer()
    lb.fit(classes)
    return (lb.classes_)

def labels2indices(labels) :
    # Process through LabelBinarizer to ensure consistency with reporting tools
    lb = LabelBinarizer()
    lb.fit(classLabels())
    return lb.transform(labels)
 

# If labels are aggregated into smaller set of labels, that should be done here
def relabel(classes) :
    #breakpoint()
    if len(classes.shape) == 2 :
        if classes.shape[1] == 1 :
            classes = np.reshape(classes,classes.shape[0])
        elif classes.shape[1] > 1 :
            classes = np.argmax(classes,axis=1)
    classes[classes == 1] = 0
    #classes[classes == 2] = 0
    classes[classes == 2] = 1
    classes[classes == 3] = 1
    #classes[classes == 4] = 1
    #classes[classes == 5] = 1
    #if np.isnumeric(classes) :
    #    classes[classes == 1] = 0
    #    classes[classes == 3] = 2
    #else :
    #    classes[classes == 'GS3-UMMCC'] = 'GS3-PANDA'
    #    classes[classes == 'GS4-UMMCC'] = 'GS4-PANDA'
    return classes

def binarizePredictions(predictions) :
    # Converting to single sigmoid from softmax probabilities
    # Note: all predictions add to 1 (softmax condition)
    # So a sum across all would yield a maximum of 1
    # If equally distributed across all classes, then we'd want a sigmoid output ~0.5.
    # If skewed far to negative class we want ~0.0, and skewed far to positive class, we want ~1.0.
    #predictions_n = -np.sum(predictions[:,0:3],axis=1)
    #predictions_p = np.sum(predictions[:,3:6],axis=1)
    predictions_n = -np.sum(predictions[:,0:2],axis=1)
    predictions_p = np.sum(predictions[:,2:4],axis=1)
    predictions_c = np.sum((predictions_n,predictions_p),axis=0)
    predictions_c = predictions_c + 1
    predictions_c = predictions_c / 2
    return predictions_c

def path2label(imagePath,relabels=None,stripPaths=False) :
    label = imagePath.split(os.path.sep)[-2]
    relabeled = 0
    if stripPaths :
        imagePath = os.path.basename(imagePath)
    if relabels is not None :
        if type(relabels) is dict :
            if imagePath in relabels :
                label = relabels[imagePath]
                relabeled = 1
        else : # presume it is simply some sort of list, set or array
            if imagePath in relabels :
                label = 'GS3' if label == 'GS4' else 'GS4'
                relabeled = 1
    # Modified the below to test independent Moffitt test set
    if label == 'GS3' :
        if 'stack' in imagePath :
            label = 'GS3-UMMCC'
        elif 'Layer' in imagePath :
            label = 'GS3-MOFFITT'
        else :
            label = 'GS3-PANDA'
    else : # GS4
        if 'stack' in imagePath :
            label = 'GS4-UMMCC'
        elif 'Layer' in imagePath :
            label = 'GS4-MOFFITT'
        else :
            label = 'GS4-PANDA'
    return (label,relabeled)

def path2subject(imagePath) :
    basepath = os.path.basename(imagePath)
    # The following can be enabled to be more performant
    ## UM/MCC specific (all UM/MCC data has "stack" in the name due to the .vsi image format)
    #subject = basepath.split('s')[0] # s for "stack"
    ## PANDA Radboud specific
    #subject = basepath.split('_')[0]
    ## PANDA or UM/MCC
    if basepath != None and 'stack' in basepath :
        subject = basepath.split('s')[0] # s for "stack"
    elif basepath != None and 'Layer' in basepath :
        subject = basepath.split('_')[0]
    elif basepath != None:
        subject = basepath.split('_')[0]
    else :
        subject = None
    return subject


