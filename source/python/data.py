import os
import numpy as np
import random
import cv2
import re
import itertools
from collections import defaultdict

from configurable.dataParameters import *
# Versioning specific utils
import pkg_resources as pkgs
from packaging import version as pkg_version
from platform import python_version

#tfversion = pkg_version.parse(pkgs.get_distribution('tensorflow').version)
#if tfversion < pkg_version.parse('2.9') :
#    from keras.preprocessing.image import img_to_array
#else :
#    from tensorflow.keras.utils import img_to_array

from sklearn.preprocessing import LabelBinarizer
#from tensorflow.keras.utils import to_categorical
from tensorflow import keras


def parseListWithDatadir(filepath,prepend_path) :
    blocklist=[]
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines :
        line = line.lstrip().rstrip()
        #if not line and not line.startswith('#') :
        if line and not line.startswith('#') :
            line = os.path.join(prepend_path,line)
            blocklist.append(line)
    return blocklist


def parseBlocklist(filepath,trimpath=False) :
    blocklist=[]
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines :
        line = line.lstrip().rstrip()
        #if not line and not line.startswith('#') :
        if line and not line.startswith('#') :
            if trimpath: line = os.path.basename(line)
            blocklist.append(line)
    return blocklist


def readMask(impath) :
    mask=cv2.imread(impath)
    img_channels = len(mask.shape)
    if(img_channels == 3) : mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask = mask.astype('float32')
    mask = np.ceil(mask/np.amax(mask))
    return mask


def applyMask(image,mask) :
    #breakpoint()
    img_channels = len(image.shape)
    minval = np.min(image)
    maskinv = mask - 1
    minmask = maskinv*minval
    if img_channels == 3 :
        image = image*mask[:,:,None]
        # This extra step is to make sure that the mask
        # is set to the darkest value in the image.
        image = image - minmask[:,:,None]
    elif img_channels == 2 :
        image = image*mask[:,:]
        image = image - minmask[:,:]
    return image


def augmentHue(image,factor) :
    imageHSV=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    vectorHSV=imageHSV.reshape((image.shape[0]*image.shape[1],3))
    h=vectorHSV[:,0]
    if image.dtype == 'uint8' :
        intfactor=int(factor*255)
        randHueShift=np.random.randint(intfactor)-int(intfactor/2)
        h=h+randHueShift
        # Note: since dtype is uint8, values will naturally wrap around in the above step
    elif image.dtype == 'uint16' :
        intfactor=int(factor*65535)
        randHueShift=np.random.randint(intfactor)-int(intfactor/2)
        h=h+randHueShift
        # Note: since dtype is uint16, values will naturally wrap around in the above step
    else : # assume it is unit normalized
        intfactor=int(factor*255)
        randHueShift=np.random.randint(intfactor)-int(intfactor/2)
        randHueShift=randHueShift/255.0
        h=h+randHueShift
        if randHueShift > 0.0 :
            h[h>1.0] = h[h>1.0] - 1.0
        else :
            h[h<0.0] = h[h<0.0] + 1.0
    vectorHSV[:,0]=h
    imageHSV=vectorHSV.reshape((image.shape[0],image.shape[1],3))
    image=cv2.cvtColor(imageHSV,cv2.COLOR_HSV2RGB)
    return image


def standardizeImage(image,normalize) :
    if normalize :
        #image = image.astype('float32')
        m=np.mean(image)
        s=np.std(image)
        image = (image - m)/s
    
    return image


class ImageProcessing :
    def __init__(self,prescale=1.0,hueAugment=None,grayed=False,normalized=True,composite="default") :
        self.prescale = prescale
        self.hueAugment = hueAugment
        self.grayed = grayed
        self.normalized = normalized
        self.composite = composite

    def __str__(self) :
        return f'Prescale:{self.prescale}, HueAugment:{self.hueAugment}, Grayed:{self.grayed}, Normalized:{self.normalized}, Composite:{self.composite}'

    def __call__(self,image) :
        if self.hueAugment is not None:
            image = augmentHue(image,self.hueAugment)
    
        image = standardizeImage(image,self.normalized)
        return image


def gradient(image) :
    if image.ndim == 3:
        gray = np.sum(image,axis=2)
    else :
        gray = image
    ddepth = cv2.CV_32F
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def loadImage(impath,maskpath,img_channels,imageProcessing) :
    #print(f'Attempting to read image: {impath}')
    if imageProcessing.grayed :
        image = cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
        if (img_channels == 3) :
            image = np.stack((image,)*3, axis=-1)
    else :
        if (img_channels == 1) :
            image = cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
        else :
            image = cv2.imread(impath)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    image = image.astype('float32')/255.0

    maskim = None
    if maskpath is not None :
        maskim = readMask(maskpath)
        if imageProcessing.composite == "default" :
            image = applyMask(image,maskim)
    # H - hematoxylin - not yet supported
    # E - eosin       - not yet supported
    # G - gradient
    # M - mask
    # h - hue
    # s - saturation
    # v - value
    # r - red
    # g - green
    # b - blue
    # B - blanked (zeros)
    if imageProcessing.composite not in {"default", None} :
        hsv = None

        numChans = len(imageProcessing.composite)
        assert numChans == img_channels, f'FATAL ERROR: composition={imageProcessing.composite}, channels={img_channels} mismatch'
        dest = np.zeros((image.shape[0],image.shape[1],numChans))
        if any(elem in imageProcessing.composite for elem in 'hsv') :
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        ## Note: ignoring directives beyond the first 3
        #if len(imageProcessing.composite)  > 3 :
        #    print(f'WARNING: ignoring composite directives beyond first 3: {imageProcessing.composite[3:]}')
        #    imageProcessing.composite = imageProcessing.composite[0:3]
        for elem in range(0,numChans) :
            letter = imageProcessing.composite[elem]
            if letter == 'G' :
                dest[:,:,elem] = gradient(image)
            elif letter == 'M' :
                if maskim is not None :
                    dest[:,:,elem] = maskim
                else :
                    dest[:,:,elem] = 0
            elif letter == 'h' :
                dest[:,:,elem] = hsv[:,:,0]
            elif letter == 's' :
                dest[:,:,elem] = hsv[:,:,1]
            elif letter == 'v' :
                dest[:,:,elem] = hsv[:,:,2]
            elif letter == 'r' :
                dest[:,:,elem] = image[:,:,0]
            elif letter == 'g' :
                dest[:,:,elem] = image[:,:,1]
            elif letter == 'b' :
                dest[:,:,elem] = image[:,:,2]
            elif letter == 'B' :
                dest[:,:,elem] = 0
            else :
                print(f'WARNING: Unknown directive ({letter}) - channel will be blanked')
                # Rewriting the directive so we only print the above message once
                imageProcesing.composite = imageProcessing.composite[0:elem] + 'B' + imageProcessing.composite[elem+1:]
        image = dest

    prescale = imageProcessing.prescale
    if 1.0 > prescale or prescale > 1.0 : # i.e. prescale != 1
        image = cv2.resize(image, (0,0), fx=prescale,fy=prescale)

    return image


def autoDownsize(image,w,h) :
    if image.shape[1] > w or image.shape[0] > h :
        scale1 = w/image.shape[1]
        scale2 = h/image.shape[0]
        scale = min([scale1,scale2])
        image = cv2.resize(image, (0,0), fx=scale,fy=scale)
    return image


def autoUpsize(image,w,h) :
    if image.shape[1] < w or image.shape[0] < h :
        scale1 = w/image.shape[1]
        scale2 = h/image.shape[0]
        scale = max([scale1,scale2])
        image = cv2.resize(image, (0,0), fx=scale,fy=scale)
    return image


def padImage(image,padded_w,padded_h):
    #assert image.shape[1] <= padded_w
    #assert image.shape[0] <= padded_h
    image = autoDownsize(image,padded_w,padded_h)
    
    border_w = padded_w - image.shape[1]
    border_h = padded_h - image.shape[0]
    left = int(border_w/2)
    right = border_w - left
    top = int(border_h/2)
    bottom = border_h - top
    image = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT)
    return image


def randomCrop(image,crop_w,crop_h):
    #assert image.shape[1] >= crop_w
    #assert image.shape[0] >= crop_h
    image = autoUpsize(image,crop_w,crop_h)
    
    max_offset_x = image.shape[1] - crop_w
    max_offset_y = image.shape[0] - crop_h
    if max_offset_x > 0 : x = np.random.randint(0, max_offset_x)
    else : x = 0
    if max_offset_y > 0 : y = np.random.randint(0, max_offset_y)
    else : y = 0
    if image.ndim == 3 :
        crop = image[y:(y + crop_h),x:(x + crop_w),:]
    else :
        crop = image[y:(y + crop_h),x:(x + crop_w)]
    return crop


def samplemix(image,crop_w,crop_h,tilerank):
    orig_crop_w = crop_w
    orig_crop_h = crop_h
    orig_tilerank = tilerank
    # This is a hack to scale down the samplemix if the source image
    # is too small.
    while image.shape[1] < crop_w or image.shape[0] < crop_h :
       crop_w = int(crop_w / 2)
       crop_h = int(crop_h / 2)
       tilerank = tilerank * 2
    assert image.shape[1] >= crop_w, f'image width({image.shape[1]}) < crop_w ({crop_w})'
    assert image.shape[0] >= crop_h, f'image height({image.shape[0]}) < crop_h ({crop_h})'
    
    max_offset_x = image.shape[1] - crop_w
    max_offset_y = image.shape[0] - crop_h
    
    x_delta = max_offset_x / (tilerank-1)
    y_delta = max_offset_y / (tilerank-1)
    
    if image.ndim == 3 :
        imsamplemix = np.zeros((orig_crop_h*orig_tilerank,orig_crop_w*orig_tilerank,image.shape[2]),dtype=image.dtype)
    else : # lif image.ndim == 2 :
        imsamplemix = np.zeros((orig_crop_h*orig_tilerank,orig_crop_w*orig_tilerank),dtype=image.dtype)
    
    for tx in range(tilerank) :
        txoffdst = int(tx*crop_w)
        txoffsrc = int(tx*x_delta)
        for ty in range(tilerank) :
            tyoffdst = int(ty*crop_h)
            tyoffsrc = int(ty*y_delta)
            imsamplemix[tyoffdst:tyoffdst+crop_h,txoffdst:txoffdst+crop_w] = image[tyoffsrc:tyoffsrc+crop_h,
                                                                              txoffsrc:txoffsrc+crop_w]
    return imsamplemix


class ImageLoader :
    def __init__(self,input_tensor_shape,imagePather,maskPather,imageProcessing,postProcessing=None) :
        _,_,self.imgChannels = input_tensor_shape
        self.imagePather = imagePather
        self.maskPather = maskPather
        self.imageProcessing = imageProcessing
        self.postProcessing = postProcessing

    def __call__(self,img_name) :
        impath = self.imagePather(img_name)
        mapath = self.maskPather(img_name)
        image = loadImage(impath,mapath,self.imgChannels,self.imageProcessing)
        if self.postProcessing:
            image = self.postProcessing(image)
        return image


class ReadResizedImage :
    def __init__(self,input_tensor_shape,imageProcessing) :
        self.img_rows,self.img_cols,_ = input_tensor_shape
        self.imageProcessing = imageProcessing

    def __call__(self,image) :
        image = cv2.resize(image, (self.img_rows, self.img_cols))
        if self.imageProcessing is not None:
            image = self.imageProcessing(image)
        return image


class ReadPaddedImage :
    def __init__(self,input_tensor_shape,imageProcessing) :
        self.img_rows,self.img_cols,_ = input_tensor_shape
        self.imageProcessing = imageProcessing

    def __call__(self,image) :
        image = padImage(image,self.img_cols,self.img_rows)
        if self.imageProcessing is not None:
            image = self.imageProcessing(image)
        return image


class ReadRandomCropOfImage :
    def __init__(self,input_tensor_shape,imageProcessing=None) :
        self.img_rows,self.img_cols,_ = input_tensor_shape
        self.imageProcessing = imageProcessing

    def __call__(self,image) :
        image = randomCrop(image,self.img_cols,self.img_rows)
        if self.imageProcessing is not None:
            image = self.imageProcessing(image)
        return image


class ReadSamplemixOfImage :
    def __init__(self,input_tensor_shape,imageProcessing,tilerank=3) :
        self.img_rows,self.img_cols,_ = input_tensor_shape
        self.imageProcessing = imageProcessing
        self.tilerank = tilerank

    def __call__(self,image) :
        #print(f'Presamplemix: image.shape={image.shape}, image.dtype={image.dtype}')
        image = samplemix(image,int(self.img_cols/self.tilerank),int(self.img_rows/self.tilerank),self.tilerank)
        #print(f'Postsamplemix: image.shape={image.shape}, image.dtype={image.dtype}')
        if self.imageProcessing is not None:
            image = self.imageProcessing(image)
        return image


class KeyInitializingDict(defaultdict):
    def __missing__(self, key):
        val = self[key] = self.default_factory(key)
        return val


class ImageCache :
    def __init__(self,imageLoader,imagePostProcessor=None) :
        #self.imageLoader = imageLoader
        self.cache = KeyInitializingDict(imageLoader)
        self.imagePostProcessor = imagePostProcessor

    def __call__(self,img_name) :
        image = self.cache[img_name]
        if self.imagePostProcessor is not None:
            image = self.imagePostProcessor(image)
        return image

#ImageLoader(channels,mask,imageProcessing,preprocess)

class ImagePath :
    def __init__(self,image_dir) :
        self.image_dir = image_dir

    def __call__(self,img_name) :
        # Note path.join will simply return img_name if it starts from "root"
        impath = os.path.join(self.image_dir,img_name)
        return impath


class MaskPath :
    def __init__(self,data_dir,mask_dir,mask_ext=None) :
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.mask_ext = mask_ext

    def __call__(self,img_name) :
        mask_name=None
        if self.mask_dir is not None :
            if self.mask_ext is not None :
                prefix, ext = os.path.splitext(img_name)
                mask_name=prefix + self.mask_ext
            else : mask_name = img_name
            # Strip data_dir path if it was added
            if mask_name.startswith(self.data_dir) :
                mask_name = './' + "".join(mask_name.split(self.data_dir))
            mask_name=os.path.join(self.mask_dir,mask_name)
            mask_name = os.path.normpath(mask_name)  
        elif self.mask_ext is not None :
            prefix, ext = os.path.splitext(img_name)
            mask_name=prefix + self.mask_ext
        return mask_name


class ResizedGenerator :
    def __init__(self,image_dir,input_tensor_shape=(224,224,3),imageProcessing=None,mask_dir=None,mask_ext=None) :
        imageProcessing = imageProcessing if imageProcessing is not None else ImageProcessing()
        # Ensure that this type of read never needlessly prescales.
        imageProcessing.prescale=1.0
        resizer = ReadResizedImage(input_tensor_shape,imageProcessing)
        loader = ImageLoader(input_tensor_shape,ImagePath(image_dir),
                             MaskPath(image_dir,mask_dir,mask_ext),
                             imageProcessing,postProcessing=resizer)
        self.cache = ImageCache(loader)

    def __call__(self,img_name) :
        return self.cache(img_name)


class FramedGenerator :
    def __init__(self,image_dir,input_tensor_shape=(224,224,3),imageProcessing=None,prescale=1.0,hueAugment=None,
                 normalize=True,mask_dir=None,mask_ext=None) :
        imageProcessing = imageProcessing if imageProcessing is not None else ImageProcessing()
        padder = ReadPaddedImage(input_tensor_shape,imageProcessing)
        loader = ImageLoader(input_tensor_shape,ImagePath(image_dir),
                             MaskPath(image_dir,mask_dir,mask_ext),
                             imageProcessing,postProcessing=padder)
        self.cache = ImageCache(loader)

    def __call__(self,img_name) :
        return self.cache(img_name)


class CroppedGenerator :
    def __init__(self,image_dir,input_tensor_shape,imageProcessing=None,mask_dir=None,mask_ext=None) :
        imageProcessing = imageProcessing if imageProcessing is not None else ImageProcessing()
        loader = ImageLoader(input_tensor_shape,ImagePath(image_dir),
                             MaskPath(image_dir,mask_dir,mask_ext),
                             imageProcessing)
        self.cache = ImageCache(loader)
        self.postProcessor = ReadRandomCropOfImage(input_tensor_shape,imageProcessing)

    def __call__(self,img_name) :
        image = self.cache(img_name)
        return self.postProcessor(image)


class SamplemixGenerator :
    def __init__(self,image_dir,input_tensor_shape,imageProcessing=None,tilerank=3,
                 mask_dir=None,mask_ext=None) :
        imageProcessing = imageProcessing if imageProcessing is not None else ImageProcessing()
        if 1.0 > imageProcessing.prescale or imageProcessing.prescale > 1.0 :
            print(f'INFO: Prescaling images by: {imageProcessing.prescale}')
        sampleMixer = ReadSamplemixOfImage(input_tensor_shape,imageProcessing,tilerank)
        loader = ImageLoader(input_tensor_shape,ImagePath(image_dir),
                             MaskPath(image_dir,mask_dir,mask_ext),
                             imageProcessing,postProcessing=sampleMixer)
        self.cache = ImageCache(loader)

    def __call__(self,img_name) :
        return self.cache(img_name)


class DataFrameGenerator :
    def __init__(self,data_frame,data_frame_names) :
        self.df_names = data_frame_names
        self.df_train = data_frame #data_frame.drop(columns=['GleasonScore', 'RandomPatchNumber','PatchName'])

    def __call__(self,img_name) :
        #breakpoint()
        img_name=img_name.removesuffix('.tiff')
        img_name=img_name.removesuffix('.png')
        img_name=img_name.replace(',',';')
        img_name=os.path.basename(img_name)
        try :
            image = self.df_names.loc[self.df_names['PatchName'] == img_name].index[0]
            image_feat = self.df_train.iloc[[image]].to_numpy()
            return image_feat
        except IndexError :
            breakpoint()
            print(f'img_name: {img_name}')
            return None


## TODO: try to create a better multiproc sharding aware solution based on TensorFlow Dataset concepts.

def divisable(dividend,divisor) :
    return dividend % divisor == 0

def optimizeBalance(most_classes,fewest_classes,batch_size,num_classes=2,numPools=1) :
    if batch_size > fewest_classes*num_classes :
        print(f'WARNING: batch_size({batch_size}) too large for # of elements')
        batch_size = fewest_classes*num_classes
        print(f'WARNING: batch_size resized to {batch_size}')
    disprop_ratio=int(np.ceil(most_classes/fewest_classes))
    numBalancedImages=num_classes*most_classes
    saved_most_classes=most_classes
    batchIterations = (np.ceil( (numBalancedImages/float(numPools)) / float(batch_size))).astype(int)
    while disprop_ratio > numPools or not divisable(numBalancedImages,numPools*batch_size) :
        if disprop_ratio > numPools :
            numPools = disprop_ratio
            # The following prevents us from increasing both numPools
            # and most_classes in a single iteration which could lead to badness...
            most_classes=saved_most_classes
            numBalancedImages=num_classes*most_classes
        batchIterations = (np.ceil( (numBalancedImages/float(numPools)) / float(batch_size))).astype(int)
        numBalancedImages = batchIterations*numPools*batch_size
        # Note: paddedSize should be divisible by num classes
        most_classes = int(np.ceil(numBalancedImages/num_classes))
        numBalancedImages=num_classes*most_classes
        disprop_ratio=int(np.ceil(most_classes/fewest_classes))
    return (most_classes,numPools,batch_size,batchIterations)


def balancedDataset(image_files,labels,batch_size,numPools) :
    separated_images={}
    # Need to do some preprocessing to separate the images into n lists.
    #print(f'labels.shape: {labels.shape}, labels.shape[0]: {labels.shape[0]}')
    if numClasses() > 2 :
        labelNums=np.argmax(labels,axis=1)
    else :
        labelNums = np.squeeze(labels)
    for ndx in range(labels.shape[0]) :
        separated_images.setdefault(labelNums[ndx], []).append(image_files[ndx])
    num_classes = len(separated_images)
    fewest_classes=1e100;
    most_classes=0;
    # Check the min and max length for all classes
    for clist in separated_images.values() :
        if  most_classes < len(clist) :
            most_classes = len(clist)
        if  fewest_classes > len(clist) :
            fewest_classes = len(clist)
    
    #print(f'num_classes: {num_classes}')
    #print(f'most_classes: {most_classes}')
    #print(f'fewest_classes: {fewest_classes}')
    #print(f'numPools: {numPools}')
    paddedSize,numPools,batch_size,batchIterations = optimizeBalance(most_classes,fewest_classes,
                                                                     batch_size,num_classes,numPools)

    #print(f'numPools: {numPools}')
    #print(f'paddedSize: {paddedSize}')
    #print(f'batchIterations: {batchIterations}')
    # Make sure everything is shuffled
    image_lists=[]
    image_lists.append(range(int(paddedSize)))
    label_lists=[]
    label_lists.append(range(int(paddedSize)))
    for k,v in separated_images.items() :
        print(f'INFO: Label({k}) has {len(v)} images')
        label_lists.append(itertools.repeat(k))
        random.shuffle(v)
        image_lists.append(itertools.cycle(v))

    images=[]
    for t in zip(*image_lists) :
        images.extend(t[1:])
    #print(f'len(images):{len(images)}')
    labels=[]
    for t in zip(*label_lists) :
        labels.extend(t[1:])
    #print(f'len(labels):{len(labels)}')
    labels = np.array(labels)
    labels.resize(len(labels),1)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    #print(f'labels.shape:{labels.shape}')
    #breakpoint()
    return (images,labels,numPools,batch_size,batchIterations)


# TODO: try to create a better multiproc sharding aware solution based on TensorFlow Dataset concepts.
class AutoBalancingPooledDataAndLabel_Iterator(keras.utils.Sequence) :

    def unbalancedDataset(self) :
        batchIterations = (np.ceil( (len(self.original_images)/float(self.numPools)) / float(self.batch_size))).astype(int)
        if self.numPools > 1 :
            paddedSize = batchIterations*self.numPools*self.batch_size
            wrappedImages = self.original_images+self.original_images[0:paddedSize-len(self.original_images)]
            #self.images = wrappedImages
            wrappedLabels = np.concatenate((self.original_labels,self.original_labels[0:paddedSize-len(self.original_labels)]))
            #self.labels = wrappedLabels
        else :
            wrappedImages = self.original_images
            wrappedLabels = self.original_labels
        return (wrappedImages,wrappedLabels)
 

    def __init__(self,generator,image_files,labels,batch_size,numPools=1,autoBalance=True,shuffle=False,name=None,activeAssert=False) :
        self.generator = generator
        self.indz = np.array([idx for idx in range(len(image_files))]).astype(int)
        self.indz = self.indz.reshape(-1)
        self.original_images = np.array(image_files)
        self.original_labels = np.array(labels)
        #self.images = image_files
        #self.labels = labels
        self.original_batch_size = batch_size
        self.original_numPools = numPools
        self.batch_size = batch_size
        self.numPools = numPools
        self.pool = 0
        self.autoBalance = autoBalance
        # Given autobalancing, it is a little bit tricky to do true shuffling,
        # so we only support a pseudo shuffle, by shuffling the order of the batches
        # and we will also shuffle intra each batch.
        self.shuffle = shuffle
        self.name = name
        self.activeAssert = activeAssert

        # Cached imbatch and labbatch for debug purposes
        self.imbatch = None
        self.labbatch = None
        self.bdx = None
        
        # Wrap some images and labels so that we ensure we have enough data for each pool-batch if
        # not already wrapped by auto-balancing. A consequence is that the last pool and the first pool
        # will have a few of the same samples, which is perfectly fine.
        if autoBalance :
           self.images,self.labels,self.numPools,self.batch_size,batchIterations = balancedDataset(image_files,labels,batch_size,numPools)
        else :
            self.images,self.labels = self.unbalancedDataset()
            #batchIterations = (np.ceil( (len(self.images)/float(self.numPools)) / float(self.batch_size))).astype(int)
            #if self.numPools > 1 :
            #    paddedSize = batchIterations*self.numPools*self.batch_size
            #    wrappedImages = self.images+self.images[0:paddedSize-len(self.images)]
            #    self.images = wrappedImages
            #    wrappedLabels = np.concatenate((self.labels,self.labels[0:paddedSize-len(self.labels)]))
            #    self.labels = wrappedLabels
        #breakpoint()
        #self.batch_ndx = [i for i in range(batchIterations)]
    
    def getImages(self) :
        return self.images
    
    def getLabels(self) :
        return self.labels

    def reshuffleAll(self) :
        if self.shuffle:
            np.random.shuffle(self.indz)
            #print(self.indz.shape)
            #breakpoint()
            self.original_images = self.original_images[self.indz]
            self.original_labels = self.original_labels[self.indz]
            if self.autoBalance :
                self.images,self.labels,self.numPools,self.batch_size,_ = balancedDataset(self.original_images,self.original_labels,self.original_batch_size,self.original_numPools)
            else :
                self.images,self.labels = self.unbalancedDataset()

    
    def on_epoch_end(self) :
        savedPool=self.pool
        self.pool = self.pool + 1
        if self.pool >= self.numPools :
            self.pool = 0
            self.reshuffleAll()
        #else :
        #    if self.shuffle:
        #        random.shuffle(self.batch_ndx)
        if savedPool != self.pool :
            if self.name is None:
                print(f'\nSwitching data to pool: {self.pool}\n')
            else:
                print(f'\nSwitching {self.name} data to pool: {self.pool}\n')

    def __len__(self) :
        return (np.ceil( (len(self.images)/float(self.numPools)) / float(self.batch_size))).astype(int)

    def assertCorrectness(self) :
        return 0
        #if self.activeAssert :
        #    imlabs = None
        #    imlabs2 = None
        #    if self.shuffle :
        #        imlabs2 = labels2indices([path2label(imname) for imname in self.imbatch[self.bdx]])
        #        imlabs = self.labbatch[self.bdx]
        #    else :
        #        imlabs2 = labels2indices([path2label(imname) for imname in self.imbatch])
        #        imlabs = self.labbatch
        #    assert np.sum(np.array(imlabs != imlabs2,dtype=int)) == 0, 'FATAL ERROR: labels and images do not match!'
        #    if numClasses() == 2 :
        #        assert np.sum(imlabs) == (imlabs.size - np.sum(imlabs)), f'FATAL ERROR: labels are not balanced: {np.sum(imlabs)} != {imlabs.size - np.sum(imlabs)}'
    
    def __getitem__(self, idx) :
        #idx_warp = self.batch_ndx[idx]
        #idxpool=int((self.pool*self.__len__())+idx_warp)
        idxpool=int((self.pool*self.__len__())+idx)

        # Note: apparently in Python we do not have to take care if end index is larger than available?
        begx = idxpool * self.batch_size
        endx = (idxpool+1) * self.batch_size
        self.imbatch = self.images[ begx : endx ]
        self.labbatch = self.labels[ begx : endx ]
        # If "pseudo" shuffling, besides shuffling the "batch" itself (with idx_warp above),
        # we also shuffle intra each batch.
        #if self.shuffle :
        #    self.bdx = [i for i in range(len(self.imbatch))]
        #    random.shuffle(self.bdx)
        #    self.imbatch=np.array(self.imbatch)
        #    self.assertCorrectness()
        #    return (np.array([self.generator(img_name) for img_name in self.imbatch[self.bdx]]),self.labbatch[self.bdx])
        #    #return (np.array([img_name for img_name in imbatch[bdx]]),labbatch[bdx])
        #else :
        #    self.assertCorrectness()
        #    return (np.array([self.generator(img_name) for img_name in self.imbatch]),self.labbatch)
        #    #return (np.array([img_name for img_name in imbatch]),labbatch)
        self.assertCorrectness()
        return (np.array([self.generator(img_name) for img_name in self.imbatch]),self.labbatch)

    def debugBatch(self) :
        return (self.imbatch,self.labbatch,self.bdx)


# TODO: try to create a better multiproc sharding aware solution based on TensorFlow Dataset concepts.
class AutoBalancingPooledDataFrameAndLabel_Iterator(keras.utils.Sequence) :
    def __init__(self,generator,image_files,labels,batch_size=500,numPools=1,autoBalance=True,shuffle=False,name=None) :
        self.generator = generator
        if autoBalance :
           image_files,labels,numPools,batch_size,batchIterations = balancedDataset(image_files,labels,batch_size,numPools)
        self.images = image_files
        self.labels = labels
        self.batch_size = batch_size
        self.numPools = numPools
        self.pool = 0
        self.autoBalance = autoBalance
        self.name = name
        # Given autobalancing, it is a little bit tricky to do true shuffling,
        # so we only support a pseudo shuffle, by shuffling the order of the batches
        # and we will also shuffle intra each batch.
        self.shuffle = shuffle
        
        # Wrap some images and labels so that we ensure we have enough data for each pool-batch if
        # not already wrapped by auto-balancing. A consequence is that the last pool and the first pool
        # will have a few of the same samples, which is perfectly fine.
        if not autoBalance :
            batchIterations = (np.ceil( (len(self.images)/float(self.numPools)) / float(self.batch_size))).astype(int)
            if self.numPools > 1 :
                paddedSize = batchIterations*self.numPools*self.batch_size
                wrappedImages = self.images+self.images[0:paddedSize-len(self.images)]
                self.images = wrappedImages
                wrappedLabels = np.concatenate((self.labels,self.labels[0:paddedSize-len(self.labels)]))
                self.labels = wrappedLabels
        
        self.batch_ndx = [i for i in range(batchIterations)]
    
    def getImages(self) :
        return self.images
    
    def getLabels(self) :
        return self.labels
    
    def on_epoch_end(self) :
        savedPool=self.pool
        self.pool = self.pool + 1
        if self.pool >= self.numPools :
            self.pool = 0
        if savedPool != self.pool :
            print(f'\nSwitching data in {self.name} to pool: {self.pool}\n')
        if self.shuffle:
            random.shuffle(self.batch_ndx)

    def __len__(self) :
        return (np.ceil( (len(self.images)/float(self.numPools)) / float(self.batch_size))).astype(int)
    
    def __getitem__(self, idx) :
        idx_warp = self.batch_ndx[idx]
        idxpool=int((self.pool*self.__len__())+idx_warp)

        # Note: apparently in Python we do not have to take care if end index is larger than available?
        imbatch = self.images[idxpool * self.batch_size : (idxpool+1) * self.batch_size]
        labbatch = self.labels[idxpool * self.batch_size : (idxpool+1) * self.batch_size]
        # If "pseudo" shuffling, besides shuffling the "batch" itself (with idx_warp above),
        # we also shuffle intra each batch.
        #breakpoint()
        if self.shuffle :
            bdx = [i for i in range(len(imbatch))]
            random.shuffle(bdx)
            imbatch=np.array(imbatch)
            return (np.squeeze(np.array([self.generator(img_name) for img_name in imbatch[bdx]])),np.squeeze(labbatch[bdx]))
            #return (np.array([img_name for img_name in imbatch[bdx]]),labbatch[bdx])
        else :
            return (np.squeeze(np.array([self.generator(img_name) for img_name in imbatch])),np.squeeze(labbatch))
            #return (np.array([img_name for img_name in imbatch]),labbatch)


class Data_Iterator(keras.utils.Sequence) :
    def __init__(self, generator, image_files, batch_size) :
        self.generator = generator
        self.images = image_files
        self.batch_size = batch_size
      
    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        # Note: apparently in Python we do not have to take care if end index is larger than available?
        imbatch = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        return np.array([self.generator(img_name) for img_name in imbatch])


def readLabels(imagePaths,data_dir='',relabels=None,stripPaths=False) :
    # note, we first enforce that we get all class labels
    labels=[]
    numRelabeled = 0
    #breakpoint()
    if relabels is not None :
        relabels = readLists(relabels,data_dir,stripPaths=stripPaths)
    for imagePath in imagePaths:
        label,relabeled = path2label(imagePath,relabels=relabels,stripPaths=stripPaths)
        labels.append(label)
        numRelabeled += relabeled
    print(f'INFO: {numRelabeled} images were relabeled')
    
    labels = np.array(labels)
    #print(labels[0:10])
    # Generate label class indices
    numeric_labels = labels2indices(labels)
    #print(numeric_labels[0:10])
    #print(f'numeric_labels.shape: {numeric_labels.shape}')
    return (numeric_labels)


# This code loosely based on imutils library but restructured to allow
# for following along symbolic links and 4 cases were split out
# to allow for the highest efficiency when iterating a list of directories
# with the fewest checks and string processing possible.
def list_files(basePath, validExts=None, contains=None) :
    if ((contains is not None) and (validExts is not None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # if the contains string is not none and the filename does not contain
                # the supplied string, then ignore the file
                if filename.find(contains) == -1 : continue

                # determine the file extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # check to see if the file has one of the extensions specified
                if ext.endswith(validExts) :
                    # construct the path to the image and yield it
                    filePath = os.path.join(baseDir, filename)
                    yield filePath

    elif ((contains is None) and (validExts is not None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # determine the file extension of the current file
                ext = filename[filename.rfind("."):].lower()

                # check to see if the file has one of the extensions specified
                if ext.endswith(validExts) :
                    # construct the path to the image and yield it
                    filePath = os.path.join(baseDir, filename)
                    yield filePath

    elif ((contains is not None) and (validExts is None)) :
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # if the contains string is not none and the filename does not contain
                # the supplied string, then ignore the file
                if filename.find(contains) == -1 : continue

                # construct the path to the image and yield it
                filePath = os.path.join(baseDir, filename)
                yield filePath

    else : # ((contains is None) and (validExts is None))
        # loop over the directory structure
        for (baseDir, dirNames, filenames) in os.walk(basePath,followlinks=True) :
            # loop over the filenames in the current directory
            for filename in filenames :
                # construct the path to the image and yield it
                filePath = os.path.join(baseDir, filename)
                yield filePath


def list_images(basePath,image_types=(".pgm",".ppm",".jpg",".jpeg",".png",".bmp",".tif",".tiff"),contains=None) :
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def filterFilelist(filelist,filterstr=None,blocklist=None) :
    if filterstr is not None :
        filelist = [p for p in filelist if re.search(filterstr,p)]
    if blocklist is not None :
        # first convert list to set
        fileset = set(filelist)
        for b in blocklist : 
            if b in fileset :
                fileset.remove(b)
        filelist = list(fileset)
    filelist.sort()
    return filelist


def processImageDir(data_dir,filterstr,blocklist,shuffled) :
    # grab the image paths and randomly shuffle them
    print(f"[INFO] loading images from {data_dir}...")
    imagePaths = sorted(list(list_images(data_dir)))
    if blocklist is not None :
        if type(blocklist) in [list,tuple] :
            blocked=[]
            for b in blocklist :
                blocked.extend(parseListWithDatadir(b,data_dir))
            blocklist=blocked
        else :
            blocklist = parseListWithDatadir(blocklist,data_dir)
    imagePaths = filterFilelist(imagePaths,filterstr,blocklist)
    if shuffled:
        random.seed(14)
        random.shuffle(imagePaths)
    return imagePaths


def readLists(lists,data_dir,stripPaths=False) :
    combinedlist = None
    if lists is not None :
        if type(lists) in [list,tuple] :
            combinedlist = []
            for b in lists :
                combinedlist.extend(parseListWithDatadir(b,data_dir))
        else :
            combinedlist = parseListWithDatadir(lists,data_dir)
        if stripPaths :
            combinedlist = stripPathnames(combinedlist)
    return combinedlist


def processImageList(data_dir,file_list,filterstr,blocklist,shuffled) :
    # grab the image paths and randomly shuffle them
    print(f"[INFO] loading images from {data_dir}/{file_list}...")
    imagePaths = None
    with open(os.path.join(data_dir,file_list)) as f:
        lines = []
        for line in f.readlines() :
            linestripped=line.strip()
            if len(linestripped) > 0 :
                linepath = os.path.join(data_dir,linestripped)
                if os.path.exists(linepath) : # O.w. assume it is a comment or something else
                    lines.append(linepath)
        imagePaths=sorted(lines)
    blocklist = readLists(blocklist,data_dir)
    #if blocklist is not None :
    #    if type(blocklist) in [list,tuple] :
    #        blocked=[]
    #        for b in blocklist :
    #            blocked.extend(parseListWithDatadir(b,data_dir))
    #        blocklist=blocked
    #    else :
    #        blocklist = parseListWithDatadir(blocklist,data_dir)
    imagePaths = filterFilelist(imagePaths,filterstr,blocklist)
    if shuffled:
        random.seed(14)
        random.shuffle(imagePaths)
    return imagePaths


def getImagePaths(data_dir,file_list,filterstr,blocklist,shuffled) :
    if type(file_list) == list :
        imagePaths=[]
        for fl in file_list :
            imagePaths.extend(processImageList(data_dir,fl,filterstr,blocklist,shuffled))
    else :
        imagePaths = processImageList(data_dir,file_list,filterstr,blocklist,shuffled)
    return imagePaths


def getDataResizedIteratorFromListing(data_dir,file_list,filterstr=None,blocklist=None,relabels=None,shuffled=False,batch_size=32,numPools=1,autoBalance=True,
                                     input_tensor_shape=(224,224,3),imageProcessing=None,mask_dir=None,mask_ext=None,name=None,activeAssert=False) :
    imagePaths = getImagePaths(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths,data_dir,relabels=relabels)
    if imageProcessing is None :
        imageProcessing = ImageProcessing()
    generator = ResizedGenerator(data_dir,input_tensor_shape,imageProcessing,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled,name=name,activeAssert=activeAssert)

    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def getDataFramedIteratorFromListing(data_dir,file_list,filterstr=None,blocklist=None,relabels=None,shuffled=False,batch_size=32,numPools=1,autoBalance=True,
                                     input_tensor_shape=(224,224,3),imageProcessing=None,mask_dir=None,mask_ext=None,name=None,activeAssert=False) :
    imagePaths = getImagePaths(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths,data_dir,relabels=relabels)
    if imageProcessing is None :
        imageProcessing = ImageProcessing()
    generator = FramedGenerator(data_dir,input_tensor_shape,imageProcessing,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled,name=name,activeAssert=activeAssert)
    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def getDataCropIteratorFromListing(data_dir,file_list,filterstr=None,blocklist=None,relabels=None,shuffled=False,batch_size=32,numPools=1,autoBalance=True,
                                     input_tensor_shape=(224,224,3),imageProcessing=None,mask_dir=None,mask_ext=None,name=None,activeAssert=False) :
    imagePaths = getImagePaths(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths,data_dir,relabels=relabels)
    if imageProcessing is None :
        imageProcessing = ImageProcessing()
    generator = CroppedGenerator(data_dir,input_tensor_shape,imageProcessing,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled,name=name,activeAssert=activeAssert)

    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def getDataSamplemixIteratorFromListing(data_dir,file_list,filterstr=None,blocklist=None,relabels=None,shuffled=False,batch_size=32,numPools=1,autoBalance=True,
                                     input_tensor_shape=(224,224,3),tilerank=3,imageProcessing=None,mask_dir=None,mask_ext=None,name=None,activeAssert=False) :
    imagePaths = getImagePaths(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths,data_dir,relabels=relabels)
    if imageProcessing is None :
        imageProcessing = ImageProcessing()
    generator = SamplemixGenerator(data_dir,input_tensor_shape,imageProcessing,tilerank=tilerank,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled,name=name,activeAssert=activeAssert)
    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def getDataSamplemixIteratorFromDir(data_dir,filterstr=None,blocklist=None,relabels=None,shuffled=False,batch_size=30,numPools=1,autoBalance=True,
                                 input_tensor_shape=(224,224,3),tilerank=3,imageProcessing=None,mask_dir=None,mask_ext=None,name=None,activeAssert=False) :
    imagePaths = processImageDir(data_dir,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths,data_dir,relabels=relabels)
    if imageProcessing is None :
        imageProcessing = ImageProcessing()
    generator = SamplemixGenerator(data_dir,input_tensor_shape,imageProcessing,tilerank=tilerank,mask_dir=mask_dir,mask_ext=mask_ext)
    dataIterator = AutoBalancingPooledDataAndLabel_Iterator(generator, imagePaths, labels, batch_size, numPools,autoBalance=autoBalance,shuffle=shuffled,name=name,activeAssert=activeAssert)
    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())


def stripPathnames(names) :
    basenames = []
    for n in names :
        bn = os.path.basename(n)
        #bn = bn.replace(',',';')
        basenames.append(bn)
    return basenames


def baseImagePaths(imagePaths) :
    imagePathDict = dict()
    for img in imagePaths :
        baseimg = os.path.basename(img)
        baseimg = baseimg.removesuffix('.tiff')
        baseimg = baseimg.removesuffix('.png')
        baseimg = baseimg.replace(',',';')
        imagePathDict[baseimg] = img
    return imagePathDict


def dataframeListingIntersection(data_frame,data_frame_names,imagePaths) :
    #breakpoint()
    dfset = set(np.array(data_frame_names['PatchName']))
    newImagePaths=[]
    imagePathsDict = baseImagePaths(imagePaths)
    rejected=0
    included=0
    for imp in imagePathsDict.keys() :
        if imp in dfset :
            newImagePaths.append(imagePathsDict[imp])
            included += 1
        else :
            rejected += 1
    print(f'ImagePaths Rejected:{rejected}, Included:{included}')
    imagePaths = newImagePaths
    #len(newImagePaths)
    #len(imagePaths)
    imagePathsDict = baseImagePaths(imagePaths)
    ipset = set(imagePathsDict.keys())
    newRows=[]
    idx=0
    rejected=0
    included=0
    for dfi in data_frame_names['PatchName']:
        if dfi in ipset:
            newRows.append(idx)
            included += 1
        else :
            rejected += 1
        idx+=1
    print(f'ImagePaths Rejected:{rejected}, Included:{included}')
    # Before selection, ensure indices of data_frame and data_frame_names are consecutive and unique
    data_frame = data_frame.reset_index(drop=True)
    data_frame = data_frame.iloc[newRows]
    data_frame = data_frame.reset_index(drop=True)
    data_frame_names = data_frame_names.reset_index(drop=True)
    data_frame_names = data_frame_names.iloc[newRows]
    data_frame_names = data_frame_names.reset_index(drop=True)
    return (data_frame,data_frame_names,imagePaths)
 
def getDataFrameIteratorFromListing(data_frame,data_frame_names,data_dir,file_list,filterstr=None,blocklist=None,relabels=None,
                                    shuffled=False,batch_size=500,numPools=1,autoBalance=True,name=None) :
    #breakpoint()
    imagePaths = getImagePaths(data_dir,file_list,filterstr,blocklist,shuffled)
    data_frame,data_frame_names,imagePaths = dataframeListingIntersection(data_frame,data_frame_names,imagePaths)
    #imagePaths = processImageList(data_dir,file_list,filterstr,blocklist,shuffled)
    labels = readLabels(imagePaths,data_dir,relabels=relabels)
    generator = DataFrameGenerator(data_frame,data_frame_names)
    #def __init__(self,data_frame,generator,image_files,labels,batch_size=500,numPools=1,autoBalance=True,shuffle=False) :
    dataIterator = AutoBalancingPooledDataFrameAndLabel_Iterator(generator,imagePaths,labels,batch_size,numPools,autoBalance=autoBalance,shuffle=shuffled,name=name)
    return (dataIterator,dataIterator.getLabels(),dataIterator.getImages())

