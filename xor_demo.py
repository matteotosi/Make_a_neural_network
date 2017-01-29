from numpy import array
from demo import NeuralNetwork

if __name__ == "__main__":
    # Example with the XOR problem.
    print("XOR DEMO")

    # Training epochs
    epochs = 50000

    # Inputs.
    x = array([[0, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 1], [0, 0]])
    # Outputs.
    y = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # The XOR problem is a nonlinear problem.
    # A Neural Network with only one layer can't resolve it.
    # Lets prove it.

    # Defining a single layer Neural Network.
    one_layer_nn = NeuralNetwork((2, 1))
    # Fitting it to the training set.
    one_layer_nn.train(x, y, epochs)
    print("Training a single layer Neural Network...")
    # Testing it with [1, 1] example -> 1 XOR 1 = 0
    print("Testing with input: {}".format([1, 1]))
    prediction = one_layer_nn.think(array([1, 1]))[-1]
    print("Prediction: {} - Actual Target: {}".format(prediction, [0]))
    # A failure as expected
    print("Failure...\n")

    # Now, lets add one hidden layer to solve the XOR problem.

    # Defining a two layers Neural Network
    two_layers_nn = NeuralNetwork((2, 4), (4, 1))
    # Fitting it to the training set
    two_layers_nn.train(x, y, epochs)
    print("Training a two layers Neural Network...")
    # Testing it with [1, 1] example -> 1 XOR 1 = 0
    prediction = two_layers_nn.think(array([1, 1]))[-1]
    print("Testing with input: {}".format([1, 1]))
    # SUCCESS!
    print("Prediction: {} - Actual Target: {}".format(prediction, [0]))
    print("SUCCESS!")
