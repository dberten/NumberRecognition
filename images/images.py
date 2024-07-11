import matplotlib.pyplot as plt
import pandas as pd

class ImagesManager():
    def __init__(self) -> None:
        pass
    # Affiche les données matricielles en image
    def displayImages(x, y, n, label=False):
        plt.figure(figsize=(20, 2))
        for i in range(10):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(x.values[i].reshape(28,28))
            if label:
                plt.title("Digit: {}".format(y[i]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def convertImages(self, x_train, x_test):
        return x_train.astype("float32"), x_test.astype("float32")

    def reshapeImages(self, x_train, x_test):
        return x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))
    
    def normalizeImages(self, x_train, x_test):
        return x_train / 255, x_test / 255