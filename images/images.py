import matplotlib.pyplot as plt
import pandas as pd

class ImagesManager():
    def __init__(self) -> None:
        pass
   
   # Converti les données matricielles en float
    def convertImages(self, x_train, x_test):
        return x_train.astype("float32"), x_test.astype("float32")

    # Formatte les données en 28x28x1 -> matrice 3D  
    def reshapeImages(self, x_train, x_test, isFromCsvDataset):
        if isFromCsvDataset :
            return x_train.values.reshape(-1, 28, 28, 1), x_test.values.reshape(-1, 28, 28, 1)
        return x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))
    
    # Normalize les données la valeur du pixel est entre 0 et 255
    def normalizeImages(self, x_train, x_test):
        return x_train / 255, x_test / 255