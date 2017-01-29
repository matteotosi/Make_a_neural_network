from numpy import exp, array, random, dot


class NeuralNetwork:
    def __init__(self, *layers):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)
        # Defining the layers of the Neural Network.
        self.layers = array([2 * random.random((layer[0], layer[1])) - 1 for layer in layers])

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            hidden_outputs, last_output = self.think(training_set_inputs)
            # list in which the deltas will be stored
            deltas = list()

            # Output layer delta computation
            output_error = training_set_outputs - last_output
            output_delta = output_error * self.__sigmoid_derivative(last_output)
            deltas.append(output_delta)

            # Hidden layers deltas computation
            for i in range(0, len(self.layers) - 1):  # 0 to n-1 (n = output_layer already processed)
                # take the delta from the previous layer
                last_delta = deltas[i]
                # take the previous layer weights
                previous_layer = self.layers[-(i + 1)]  # previous layer position = -(i + 1)
                hidden_layer_error = last_delta.dot(previous_layer.T)
                # current layer output position = -(i + 1) NOT -(i + 2) because last output is not in.
                hidden_layer_delta = hidden_layer_error * self.__sigmoid_derivative(hidden_outputs[-(i + 1)])
                deltas.append(hidden_layer_delta)

            # Calculate how much to adjust the weights.
            # Adjustments_inputs = [training_set_inputs, layer_1_output, ... , layer_n-1_output].
            adjustment_inputs = [training_set_inputs] + hidden_outputs
            # Reversing deltas in correct order (for code simplicity).
            deltas = deltas[::-1]
            # Adjustment calculation for each layer.
            adjustments = array([adjustment_inputs[i].T.dot(deltas[i]) for i in range(len(adjustment_inputs))])

            # Adjust the weights. (numpy <3)
            self.layers += adjustments

    # The neural network thinks.
    def think(self, inputs):
        outputs = list()
        outputs.append(self.__sigmoid(dot(inputs, self.layers[0])))  # first layer output
        for i in range(1, len(self.layers)):
            outputs.append(self.__sigmoid(dot(outputs[i - 1], self.layers[i])))  # remaining layers output
        # return (hidden layers outputs, last layer output)
        return outputs[:-1], outputs[-1]


if __name__ == "__main__":
    # Initialise a multi layer neural network. (3 layers)
    # Each layer is represented by a tuple (num_inputs, num_neurons)
    neural_network = NeuralNetwork((3, 4), (4, 4), (4, 1))

    print("Random starting synaptic weights: ")
    print(neural_network.layers)

    # The training set.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("New synaptic weights after training: ")
    print(neural_network.layers)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
    _, pred = neural_network.think(array([1, 0, 0]))
    print(pred)
