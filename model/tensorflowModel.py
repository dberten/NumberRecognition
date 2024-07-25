import os
import base64
import keras
import matplotlib.pyplot as plt
import numpy as np
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

    # Compile le modèle    
    def compile(self, loss, optimizer, metrics):
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self._isCompiled = True
    
    # Entraîne le modèle
    def fit(self, x_train, y_train, epochs, batchSize, verbose, validationData):
        try:
            if not self._isCompiled:
                raise compileException(message="Neuronal network is not compiled.")
            return self._model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batchSize, validation_data=validationData)
        except Exception:
            raise fitException(message="Error during the neuronal network training.")

    # Ajoute une couche au réseau    
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

    # Evaluel l'efficacité du modèle (perte & prédiction)
    def evaluate(self, x_train, y_train, verbose):
        try:
            if not self._isCompiled:
                raise compileException(message="Neuronal network is not compiled.")
            else:
                return self._model.evaluate(x=x_train, y=y_train, verbose=verbose)
        except Exception:
            raise evaluateException(message="Failed to evaluate the neuronal network.")
    
    # Sauvegarde du modèle
    def save(self):
        try:
            if self._isCompiled:
                path = self._root + "/model.keras"
                self._model.save(path)
        except Exception:
            raise saveException(message="This model can't be saved.")
    
    # Donne la prédiction du modèle
    def predict(self, value):
        if self._isCompiled:
            return self._model.predict(value)
        else:
            raise compileException(message="Neuronal network is not compiled.")

    # Affiche la prédiction du modèle
    def displayPrediction(self, images, labels, predictions, nb_rows, nb_cols):
        fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                img = images[i].reshape(28, 28)
                true_label = labels[i]
                predicted_label = np.argmax(predictions[i])
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Label: {true_label}, Pred: {predicted_label}')
                ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Affiche le graph de perte et de précision
    def displayGraphLoosAcc(self, history):
        # Graph de perte
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['val_loss'], label='Valeur Perte')
        plt.title('Graphique représentant la perte')
        plt.xlabel('Epoch')
        plt.ylabel('Perte')
        plt.legend()
        # Graph de précision
        plt.subplot(1, 2, 2)
        plt.plot(history.history['val_accuracy'], label='Valeur Précision')
        plt.title('Graphique représentant la précision')
        plt.xlabel('Epoch')
        plt.ylabel('Précision')
        plt.legend()
        plt.show()