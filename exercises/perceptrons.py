import numpy as np
from tensorflow import keras
from data_loader import load_mnist_dataset
import pprint
pp = pprint.PrettyPrinter(indent=6)

class Perceptron():
    def __init__(self, train_set, validation_set, test_set):
        train_x, _ = train_set
        img_shape = np.expand_dims(train_x.shape[1:], axis=-1)
        # pp.pprint(img_shape)
        # print(type(img_shape))
        num_labels = 10
        self.network = self.construct_network(
            input_shape=img_shape, 
            num_output=num_labels)
        sgd = keras.optimizers.SGD(lr=1e-4)
        self.model = self.compile_network(
            self.network, 
            train_set, validation_set, test_set, 
            sgd)

    def __str__(self):
        return "Hello World, I am a simple perceptron."

    # This method is independent of class-instance state and is intended for class level utility
    # And with this method, it does not interact with other class-instances.
    
    @staticmethod
    def construct_network(input_shape, num_output):
        network = keras.models.Sequential()

        network.add(keras.layers.InputLayer(input_shape=input_shape))
        network.add(keras.layers.Flatten())
        network.add(keras.layers.Dense(num_output, activation='softmax'))

        return network
    
    @staticmethod
    def compile_network(network, optimizer):
        network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train_model(self, train, validation, batch_size, epochs, verbose=1):
        train_x, train_y = train
        log = self.model.fit(train_x, train_y, batch_size, epochs, verbose, validation_data=validation)
        return log

if __name__=='__main__':
    train, validation, test = \
            load_mnist_dataset(num_validation=1000)

    my_perceptron = Perceptron(train_set=train, validation_set=validation, test_set=test)
    log = my_perceptron.train_model(train, validation, 16, 20, 1)

