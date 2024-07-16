import matplotlib.pyplot as plt
import pandas as pd

class ImagesManager():
    def __init__(self) -> None:
        pass
   
    def convertImages(self, x_train, x_test):
        return x_train.astype("float32"), x_test.astype("float32")

    def reshapeImages(self, x_train, x_test, isFromCsvDataset):
        if isFromCsvDataset :
            return x_train.values.reshape(-1, 28, 28, 1), x_test.values.reshape(-1, 28, 28, 1)
        return x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))
    
    def normalizeImages(self, x_train, x_test):
        return x_train / 255, x_test / 255