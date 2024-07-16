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
    train_path = "bin/train.csv"
    test_path = "bin/test.csv"
    csvManager = CsvManager(path=train_path)
    x_train, y_train = "", ""
    x_test, y_test = "", ""
    images = ImagesManager()
    isFromCsvDataset = False
    if os.path.exists(train_path):
        train = csvManager.read_csv(csvManager.path)
        x_train = csvManager.drop_csv(train, column="label")
        y_train = train["label"]
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
        isFromCsvDataset = True
    else:
        (x_train, y_train), (x_test, y_test) = DataSet().LoadData()
        print(x_train.shape)
    x_train, x_test = images.reshapeImages(x_train, x_test, isFromCsvDataset)
    x_train, x_test = images.convertImages(x_train, x_test)
    x_train, x_test = images.normalizeImages(x_train, x_test)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # instanciation du réseau neuronal
    model = TensorFlowModel()
    if not os.path.exists("bin/model.keras"):
        model.add(type=layerEnum.CONV, layerParameters={"unitSize" : 32, "kernelSize" : (3, 3), "activation" : "relu"})
        model.add(type=layerEnum.DENSE, layerParameters={"unitSize" : 128, "activation" : "relu"})
        model.add(type=layerEnum.MAXPOOLING, layerParameters={"poolSize" : (2, 2)})
        model.add(type=layerEnum.DROPOUT, layerParameters={"unitSize" : 0.25})
        model.add(type=layerEnum.FLATTEN, layerParameters=None)
        model.add(type=layerEnum.DROPOUT, layerParameters={"unitSize" : 0.5})
        model.add(type=layerEnum.DENSE, layerParameters={"unitSize" : 10, "activation" : "softmax"})
        model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=["accuracy"])
    model.fit(x_train=x_train, y_train=y_train, epochs=10, batchSize=128, verbose=2, validationData=(x_test, y_test))
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    model_base64 = model.save()