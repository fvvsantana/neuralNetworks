import numpy as np
from inc.neuron import Neuron
import inc.activationFunctions as af

class Adaline(Neuron):
    def __init__(self, weights = None, bias = None):
        super().__init__(weights, bias, af.linear)

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
