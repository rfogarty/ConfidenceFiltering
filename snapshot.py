import math
import os
from keras.callbacks import Callback
from keras import backend

# The SnapshotEnsemble class is partially attributed to Jason Brownlee as described in his blog reference below.
# However, this code has been significantly modified from his version for robustness. In particular,
# the following significant features have been added:
#
#    1) cosine-annealed learning rate avoids dropping to zero, which creates pathologic issues,
#    2) adds a warmup period where training does not begin cosine-annealing until after some 
#       number of epochs (default=5) have run.
#
# Currently, snapshots are saved off in the HDF5 keras format.
#
# Reference: https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/
class SnapshotEnsemble(Callback):
    # constructor
    def __init__(self, filepath, iterationsPerCycle, lrate_max, lr_max_reduction_rate = 0, warmupEpochs=5, verbose=False):
        self.filepath = filepath
        self.iterationsPerCycle = iterationsPerCycle
        self.lr_max = lrate_max
        self.lr_max_reduction_rate = lr_max_reduction_rate
        self.currLR = lrate_max
        self.warmupEpochs = warmupEpochs
        self.verbose = verbose
        self.lrates = list()
        self.iteration = 0
        self.snapshot = 0
        self.lrLogName = os.path.splitext(f'logLearningRate{filepath}')[0]
        snapshot_epoch = 0
        self.lrLogName = f'{self.lrLogName.format(**locals())}.txt'

    
    def logLR(self,val) :
        with open(self.lrLogName,'a') as lrlog :
            lrlog.write(f'{val}\n')
 
    # calculate learning rate for iteration (within epoch)
    def cosine_annealing(self, iteration):
        if self.warmupEpochs > 0 :
            self.logLR(self.lr_max)
            self.currLR = self.lr_max
            return self.lr_max
        else :
            cos_inner = (math.pi * iteration) / max(self.iterationsPerCycle,iteration+1)
            alpha = 0.02 # Value dips to ~2% of the original value.
            scale = (math.cos(cos_inner) + 1.0)/2.0
            scale = (1.0-alpha)*scale + alpha
            lr = self.lr_max * scale
            self.logLR(lr)
            self.currLR = lr
            return lr


    def on_epoch_begin(self,epoch,logs=None) :
        # log value
        self.lrates.append(self.currLR)


    # calculate and set learning rate at the start of each iteration or batch
    def on_train_batch_begin(self, batch, logs=None):
        # calculate learning rate
        lr = self.cosine_annealing(self.iteration)
        #print(f'batch={batch}, iteration={self.iteration}, lr={lr}')
        if self.warmupEpochs == 0 :
            self.iteration += 1
        # set learning rate
        self.model.optimizer.learning_rate = lr
        #backend.set_value(self.model.optimizer.lr, lr)


    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        # First decrement warmupEpochs if necessary
        if self.warmupEpochs > 0 :
            self.warmupEpochs -= 1
            if self.verbose:
                print(f'>no snapshot saved at epoch {epoch}')
        # check if we can save model
        elif self.iteration >= self.iterationsPerCycle:
            # save model to file
            snapshot_epoch = self.snapshot
            self.snapshot += 1
            filename = self.filepath.format(**locals())
            if self.verbose: print(f'>Attempting to save snapshot {filename}, on epoch {epoch}')
            self.model.save(filename)
            print(f'>saved snapshot {filename}, on epoch {epoch}')
            self.iteration=0
        elif self.verbose:
            print(f'>no snapshot saved at epoch {epoch}')
        if self.lr_max_reduction_rate > 0 :
            # Some important formulas:
            #    endval = m.exp(n*m.log(-rate+1)+m.log(begval))
            #    rate = -1 * ( ((endval/begval)**(1/n)) - 1 ) # extra parantheses added for clarity
            self.lr_max = self.lr_max - self.lr_max * self.lr_max_reduction_rate
            print


