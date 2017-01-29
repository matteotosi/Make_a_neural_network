# Make_a_neural_network
This is the code for the challenge "Make a Neural Network" - Intro to Deep Learning #2 by Siraj Raval on Youtube

##Challenge description
The challenge for this video is to create a 3 layer feedforward neural network using only numpy as your dependency.

In ``demo.py`` the class ``NeuralNetwork`` of the base example has been updated with a more general implementation,
in which it's possible to define an arbitrary number of layers.

For example, the code for defining a 3 layers feedforward neural network is:

``neural_network = NeuralNetwork((3, 4), (4, 4), (4, 1))``

Where each tuple in the constructor represents a single layer (num_inputs, num_neurons).

The implementation is made with python 3.

##Usage
In a terminal run:

``python demo.py`` -> 3 layer neural network in action on the same original example.

``python xor_demo.py`` -> Solving the XOR problem, the simplest nonlinear problem.

``python iris_demo.py`` -> Example of usage of a multi layer perceptron (6 layers) on the iris dataset.

##Dependencies

numpy

scikit-learn (only for the iris dataset loading)