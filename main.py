import tensorflow as tf
from dataset import dataset

if __name__ == "__main__":
    data = dataset.DataSet().LoadData()
    print(data)
    print("hello")