import tensorflow as tf
from dataset import dataset
from file import csv

if __name__ == "__main__":
    train_path = "bin/train.csv"
    test_path = "bin/test.csv"
    data = dataset.DataSet().LoadData()
    csvManager = csv.CsvManager(path=train_path)
    train = csvManager.read_csv(csvManager.path)
    # train = csvManager.drop_csv(train, column="label")
    train = csvManager.head_csv(file=train)
    print(train)
    #print(data)
    print("hello")