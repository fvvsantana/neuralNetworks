import pandas as pd
import random as rd
import inc.activationFunctions as af
from inc.perceptron import Perceptron

'''
    This is a really dumb training where it generates random weights and biases
    and picks the best combination.

'''
def trainNeuron(df, maxError = 0.4, maxIter = 1000):
    neuron = Perceptron()
    neuron.af = af.linear
    acc = 0
    iter = 0
    range = 5
    while (1 - acc > maxError) and (iter < maxIter):
        # Generate random weights and biases:
        neuron.weights = (rd.uniform(-range, range), rd.uniform(-range,range),
                        rd.uniform(-range, range), rd.uniform(-range,range))
        neuron.bias = rd.uniform(-range, range)

        # Calculate accuracy
        acc = calculateAccuracy(df, neuron)
        iter += 1

    return {'weights': tuple(neuron.weights), 'bias': neuron.bias, 'acc': acc, 'iter': iter}

# Calculate accuracy of neuron classification over the dataset
def calculateAccuracy(df, neuron):
    acum = 0
    for index, row in df.iterrows():
        inputs = (row.V1, row.V2, row.V3, row.V4)
        classification = classifyFromNeuron(inputs, neuron)
        #print(classification)

        trueClass = row.V5
        if trueClass == (classification + 1):
            acum += 1
    return acum/len(df)

# Auxiliar function to classify inputs using a single neuron
def classifyFromNeuron(inputs, neuron):
    neuronOutput = neuron.run(inputs)
    if neuronOutput > 0.5:
        return 1
    else:
        return 0

def main():
    # Read csv file to dataframe
    df = pd.read_csv("Aula2-exec2.csv")

    # Find good parameters
    result = trainNeuron(df, 0.005)

    # Print results
    print('Accuracy:', result['acc'])
    print('Number of iterations:', result['iter'])
    print('Weights:', result['weights'])
    print('Bias:', result['bias'])


if __name__ == '__main__':
    main()
