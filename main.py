import tensorflow as tf
import os
from dataset.dataset import DataSet
from file.csv import CsvManager
from images.images import ImagesManager
from model.tensorflowModel import TensorFlowModel
from model.enumeration.layerEnum import layerEnum
from keras.src.losses import categorical_crossentropy
from keras.src.optimizers import Adadelta
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == "__main__":

    # Instanciation du DataSet
    dataSet = DataSet()
    (x_train, y_train), (x_test, y_test) = dataSet.getDataSet()
    labels = y_train
    isFromCsvDataset = os.path.exists("bin/train.csv")
    print("Is this from csv file ? ", isFromCsvDataset)
    images = ImagesManager()
    print("Hello Big Brain you have to find : " + ', '.join(map(str, y_train[:9])))

    # Mise en forme du Data Set
    x_train, x_test = images.reshapeImages(x_train, x_test, isFromCsvDataset)
    x_train, x_test = images.convertImages(x_train, x_test)
    x_train, x_test = images.normalizeImages(x_train, x_test)
    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)

    # Instanciation du réseau neuronal
    model = TensorFlowModel()
    if not os.path.exists("bin/model.keras"):
        print("No model found.")
        model.add(type=layerEnum.CONV, layerParameters={"unitSize" : 32, "kernelSize" : (3, 3), "activation" : "relu"})
        model.add(type=layerEnum.DENSE, layerParameters={"unitSize" : 128, "activation" : "relu"})
        model.add(type=layerEnum.MAXPOOLING, layerParameters={"poolSize" : (2, 2)})
        model.add(type=layerEnum.DROPOUT, layerParameters={"unitSize" : 0.25})
        model.add(type=layerEnum.FLATTEN, layerParameters=None)
        model.add(type=layerEnum.DROPOUT, layerParameters={"unitSize" : 0.5})
        model.add(type=layerEnum.DENSE, layerParameters={"unitSize" : 10, "activation" : "softmax"})
        model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=["accuracy"])
    hist = model.fit(x_train=x_train, y_train=y_train, epochs=10, batchSize=128, verbose=2, validationData=(x_test, y_test))
    model.save()

    # Interprétation des résultats du modèle
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Accuracy : {accuracy * 100} %")
    predictions = model.predict(x_train)
    model.displayPrediction(x_train[:9], labels[:9], predictions=predictions[:9], nb_rows=3, nb_cols=3)
    model.displayGraphLoosAcc(history=hist)