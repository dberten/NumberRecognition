import pytest
from keras import datasets

def test_load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)