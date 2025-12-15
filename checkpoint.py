from keras.callbacks import Callback

class LossPlusValidationLoss :
    def __init__(self,loss='loss',validationLoss='val_loss') :
        self.loss = loss
        self.validationLoss = validationLoss

    def __call__(self,logs,oldPerf) :
        newPerf = logs[self.loss] + logs[self.validationLoss]
        isBest = True
        if oldPerf is not None and oldPerf < newPerf:
            isBest = False
        return (isBest,newPerf)


class ProgrammableModelCheckpoint(Callback):
    # constructor
    def __init__(self, filepath, monitorFunc,save_weights_only=False,initBest=None,verbose=False):
        self.filepath = filepath
        self.monitorFunc = monitorFunc
        self.saveWeightsOnly = save_weights_only
        self.bestSoFar = initBest
        self.verbose = verbose
    

    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        isBest,newPerf = self.monitorFunc(logs,self.bestSoFar)
        if isBest :
            self.bestSoFar = newPerf
            filename = self.filepath.format(**locals())
            if self.saveWeightsOnly :
                if self.verbose :
                    print(f'Model weights saved as {filename}, on epoch {epoch}')
                self.model.save_weights(filename)
            else :
                if self.verbose :
                    print(f'Full Model saved as {filename}, on epoch {epoch}')
                self.model.save(filename)
        elif self.verbose :
            print(f'Model has not improved on epoch {epoch} - model not written')


