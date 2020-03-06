import numpy as np

class Perceptron:
    def __init__(self, weights = None, bias = None, activationFunc = None):
        self.weights = weights
        self.bias = bias
        self.af = activationFunc

    def run(self, inputs):
        # Put bias to arrays
        weights = list(self.weights)
        weights.insert(0, 1)
        inputs = list(inputs)
        inputs.insert(0, self.bias)

        # Scalar product of arrays
        scalarProduct = np.dot(weights, inputs)

        # Apply activation function
        return self.af(scalarProduct)
