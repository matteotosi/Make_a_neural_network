from demo import NeuralNetwork
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    # Example with the iris dataset, loaded with sklearn
    print("IRIS DEMO")

    # Training epochs
    epochs = 50000

    # One-Hot encoder for encoding labels
    enc = OneHotEncoder()

    # Loading iris dataset
    iris = load_iris()

    # Defining training inputs and outputs
    x = iris.data[:-1]
    y = iris.target[:-1]
    y = y.reshape(len(y), 1)
    y = enc.fit_transform(y).toarray()

    print("Training input shape: {}".format(x.shape))
    print("Training output shape: {}".format(y.shape))

    # Pickling one example for testing
    test_x = iris.data[-1]
    test_y = iris.target[-1]
    test_y = enc.transform(iris.target[-1]).toarray()

    # Building a Neural Network with 6 hidden layers
    neural_network = NeuralNetwork((x.shape[1], 10), (10, 10), (10, 10), (10, 10), (10, 10), (10, y.shape[1]))
    # Training the Neural Network
    print("Started training with {} epochs...".format(epochs))
    neural_network.train(x, y, epochs)

    # Testing the Neural Network
    pred = np.array(neural_network.think([test_x])[-1])
    print("Test input: {}".format(test_x))
    if pred.argmax() == test_y.argmax():
        print("SUCCESS! predicted label: {} - actual label: {}".format(pred.argmax(), test_y.argmax()))
    else:
        print("Failure... predicted label: {} - actual label: {}".format(pred.argmax(), test_y.argmax()))
        print("GO DEEPER!")
