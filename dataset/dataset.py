import keras.src.datasets.mnist as mnist
import tensorflow as tf
import os
from file.csv import CsvManager
from images.images import ImagesManager
from sklearn.model_selection import train_test_split

class DataSet():
    _mnistData = mnist.load_data()
    _x_train = ""
    _y_train = ""
    _x_test = ""
    _y_test = ""

    def __init__(self):
        pass

    def loadDataFromKeras(self):
        (x_train, y_train), (x_test, y_test) = self._mnistData
        return (x_train, y_train), (x_test, y_test)
    
    def loadDataFromCsv(self, csvManager: CsvManager):
        train = csvManager.read_csv(csvManager.path)
        x_train = csvManager.drop_csv(train, column="label")
        y_train = train["label"]
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
        return (x_train, y_train), (x_test, y_test)
    
    def getDataSet(self):
        train_path = "bin/train.csv"
        csvManager = CsvManager(path=train_path)
        if os.path.exists(train_path):
            (self._x_train, self._y_train), (self._x_test, self._y_test) = self.loadDataFromCsv(csvManager)
        else:
            (self._x_train, self._y_train), (self._x_test, self._y_test) = self.loadDataFromKeras()
        return (self._x_train, self._y_train), (self._x_test, self._y_test)