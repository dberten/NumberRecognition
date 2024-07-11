import tensorflow
import keras
from exception.tensorFlow.layerException import layerException
from exception.tensorFlow.fitException import fitException
from exception.tensorFlow.compileException import compileException
from exception.tensorFlow.evaluateException import evaluateException
from keras.src.models import Sequential
from keras.src.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.src import Input
from model.enumeration.layerEnum import layerEnum

class TensorFlowModel():
    def __init__(self):
        self._model = Sequential([Input(shape=(28, 28, 1))])
        self._isCompiled = False
        
    def compile(self, loss, optimizer, metrics):
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self._isCompiled = True
    
    def fit(self, x_train, y_train, epochs, batchSize, verbose, validationData):
        try:
            if not self._isCompiled:
                raise compileException(message="Neuronal network is not compiled.")
            else:
                self._model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batchSize, validation_data=validationData)
        except Exception:
            raise fitException(message="Error during the neuronal network training.")
        
    def add(self, type, layerParameters):
        if type == layerEnum.DENSE:
            self._model.add(Dense(layerParameters["unitSize"], activation=layerParameters["activation"]))
        elif type == layerEnum.CONV:
            self._model.add(Conv2D(layerParameters["unitSize"], kernel_size=layerParameters["kernelSize"]))
        elif type == layerEnum.MAXPOOLING:
            self._model.add(MaxPooling2D(pool_size=layerParameters["poolSize"]))
        elif type == layerEnum.DROPOUT:
            self._model.add(Dropout(layerParameters["unitSize"]))
        elif type == layerEnum.FLATTEN:
            self._model.add(Flatten())
        else:
            raise layerException(message="Layer doesn't exist.")

    def evaluate(self, x_train, y_train, verbose):
        try:
            if not self._isCompiled:
                raise compileException(message="Neuronal network is not compiled.")
            else:
                return self._model.evaluate(x=x_train, y=y_train, verbose=verbose)
        except Exception:
            raise evaluateException(message="Failed to evaluate the neuronal network.")