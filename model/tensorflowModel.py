import keras.src
import tensorflow
import os
import base64
import keras
from exception.tensorFlow.layerException import layerException
from exception.tensorFlow.fitException import fitException
from exception.tensorFlow.compileException import compileException
from exception.tensorFlow.evaluateException import evaluateException
from exception.tensorFlow.saveException import saveException
from keras.src.models import Sequential
from keras.src.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.src import Input
from keras.src.saving import load_model
from model.enumeration.layerEnum import layerEnum

class TensorFlowModel():
    _root = "bin"

    def __init__(self):
        if os.path.exists(self._root + "/model.keras"):
            self._model = load_model(self._root + "/model.keras")
            self._isCompiled = True
        else:
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
    
    def save(self):
        try:
            if self._isCompiled:
                path = self._root + "/model.keras"
                self._model.save(path)
                with open(path, "rb") as file:
                    model_bin = file.read()
                model_base64 = base64.b64encode(model_bin)
                return model_base64
        except Exception:
            raise saveException(message="This model can't be saved.")