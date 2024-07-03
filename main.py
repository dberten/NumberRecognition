import tensorflow as tf
import os
from dataset import dataset
from file import csv
from images import images

if __name__ == "__main__":
    train_path = "bin/train.csv"
    test_path = "bin/test.csv"
    csvManager = csv.CsvManager(path=train_path)
    if os.path.exists(train_path):
        train = csvManager.read_csv(csvManager.path)
        # train = csvManager.drop_csv(train, column="label")
        train = csvManager.head_csv(file=train)
        imagesToDisplay = images.ImagesManager()
        imagesToDisplay.displayImages(csvManager.drop_csv(train, column="label"), train["label"], n=10, label=True)
        print(train)
    else:
        data = dataset.DataSet().LoadData()
        # (x_train, y_train), (x_test, y_test) = data
        print(data)

    #print(data)
    print("hello")