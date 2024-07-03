import matplotlib.pyplot as plt
import pandas as pd

class ImagesManager():
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

    def convertImages(x, y):
        return x.astype("float32"), y.astype("float32")

    def reshapeImages(x, y):
        return x.reshape(60000, 784), y.reshape(10000, 784)
    
    def normalizeImages(x, y):
        return x / 255, y /255