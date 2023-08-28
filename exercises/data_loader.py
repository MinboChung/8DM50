from tensorflow import keras
from copy import deepcopy
import numpy as np

def load_mnist_dataset(num_validation):
    # The images are 28x28, and 10 labels
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

    # Convert train_y to one-hot encoding
    train_y = keras.utils.to_categorical(train_y, 10)
    test_y  = keras.utils.to_categorical(test_y, 10)

    # Divide the train set into train and validation set
    validation_x = deepcopy(train_x[:num_validation])
    validation_y = deepcopy(train_y[:num_validation])
    train_x = train_x[num_validation:]
    train_y = train_y[num_validation:]

    # Add a new axis for greyscale value, it is purposed for feeding with CNN's expectation.
    train_x = np.expand_dims(train_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)
    validation_x = np.expand_dims(validation_x, axis=-1)

    """
        If the images were RGB then, we have to assign x = np.repeat(x, 3, axis=-1)
    """
    return (train_x, train_y), (validation_x, validation_y), (test_x, test_y)