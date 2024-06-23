import keras.datasets.mnist as mnist
import tensorflow as tf

class DataSet():
    _mnistData = mnist.load_data()
    def LoadData(self):
        (x_train, y_train), (x_test, y_test) = self._mnistData
        return x_train, y_train, x_test, y_test