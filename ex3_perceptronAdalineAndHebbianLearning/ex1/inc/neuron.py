from abc import ABC, abstractmethod
import numpy as np

class Neuron(ABC):
    def __init__(self, weights = None, bias = None, activationFunc = None):
        self.weights = weights
        self.bias = bias
        self.af = activationFunc

    def run(self, inputs):
        pass
