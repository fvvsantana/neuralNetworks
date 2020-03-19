import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns

import inc.activationFunctions as af
from inc.perceptron import Perceptron
from inc.adaline import Adaline

def trainPerceptron(df, minLearningRate = 0.005, maxIter = 100):
    neuron = Perceptron()
    acc = 0
    iter = 0
    learningRate = 1
    lastAccuracies = list()
    for i in range(10):
        lastAccuracies.append(0)
        lastAccuracies.append(1)

    # Track error during training
    errorEvolution = list()

    # Generate random weights and biases:
    radius = 5
    neuron.weights = list()
    for i in range(df.shape[1] - 1):
        neuron.weights.append(rd.uniform(-radius, radius))

                    #rd.uniform(-radius, radius), rd.uniform(-radius,radius)]
    neuron.bias = rd.uniform(-radius, radius)

    while (learningRate > minLearningRate) and (iter < maxIter):
        # Loop over dataset once, update weights and return accuracy
        acc = iteratePerceptronTraining(df, neuron)

        # Update learningRate
        del lastAccuracies[0]
        lastAccuracies.append(acc)
        learningRate = max(lastAccuracies) - min(lastAccuracies)

        # Update error
        errorEvolution.append(1-acc)
        # Update iteration counter
        iter += 1

    return neuron, errorEvolution

# Loop over dataset once, update weights and return accuracy
def iteratePerceptronTraining(df, neuron, weightUpdateStep = 1):
    hits = 0
    for index, row in df.iterrows():
        # Classify instance
        inputs = row[0:-1]
        neuronOutput = neuron.run(inputs)
        if neuronOutput > 0.5:
            classification = 2
        else:
            classification = 1

        # Update weights
        trueClass = row[-1]
        for i in range(len(neuron.weights)):
            neuron.weights[i] += weightUpdateStep * inputs[i] * (trueClass - classification)
        neuron.bias += weightUpdateStep * (trueClass - classification)


        # Update hits
        if trueClass == classification:
            hits += 1

    return hits/len(df)

def trainAdaline(df, minLearningRate = 0.005, maxIter = 100):
    neuron = Adaline()
    acc = 0
    iter = 0
    learningRate = 1
    lastAccuracies = list()
    for i in range(10):
        lastAccuracies.append(0)
        lastAccuracies.append(1)

    # Track error during training
    errorEvolution = list()

    # Generate random weights and biases:
    radius = 5
    neuron.weights = list()
    for i in range(df.shape[1] - 1):
        neuron.weights.append(rd.uniform(-radius, radius))

    neuron.bias = rd.uniform(-radius, radius)

    while (learningRate > minLearningRate) and (iter < maxIter):
        # Loop over dataset once, update weights and return accuracy
        acc = iterateAdalineTraining(df, neuron)

        # Update learningRate
        del lastAccuracies[0]
        lastAccuracies.append(acc)
        learningRate = max(lastAccuracies) - min(lastAccuracies)

        # Update error
        errorEvolution.append(1-acc)
        # Update iteration counter
        iter += 1

    return neuron, errorEvolution

# Loop over dataset once, update weights and return accuracy
def iterateAdalineTraining(df, neuron, weightUpdateStep = 1):
    hits = 0
    for index, row in df.iterrows():
        # Classify instance
        inputs = row[0:-1]
        neuronOutput = neuron.run(inputs)
        if neuronOutput > 0.5:
            classification = 2
        else:
            classification = 1

        # Update weights
        trueClass = row[-1]
        for i in range(len(neuron.weights)):
            neuron.weights[i] += weightUpdateStep * inputs[i] * (trueClass - (neuronOutput + 1))
        neuron.bias += weightUpdateStep * (trueClass - classification)

        # Update hits
        if trueClass == classification:
            hits += 1

    return hits/len(df)


def main():
    # Dataset 1
    # Read csv file to dataframe
    df = pd.read_csv('../data/Aula3-dataset_1.csv')

    sns.pairplot(df, hue=df.columns[-1])
    plt.title('Dataset1 - True')
    plt.show(block=True)

    # Perceptron
    # Training
    neuron, errorEvolution = trainPerceptron(df)

    # Plot error evolution
    plt.plot(errorEvolution)
    plt.xlabel('Iteration')
    plt.ylabel('Relative error')
    plt.title('Dataset1 - Perceptron')
    plt.show(block=True)



    # Adaline
    # Training
    neuron, errorEvolution = trainAdaline(df)

    # Plot error evolution
    plt.plot(errorEvolution)
    plt.xlabel('Iteration')
    plt.ylabel('Relative error')
    plt.title('Dataset1 - Adaline')
    plt.show(block=True)

    # Dataset 2
    # Read csv file to dataframe
    df = pd.read_csv('../data/Aula3-dataset_2.csv')

    # Perceptron
    # Training
    neuron, errorEvolution = trainPerceptron(df)

    # Plot error evolution
    plt.plot(errorEvolution)
    plt.xlabel('Iteration')
    plt.ylabel('Relative error')
    plt.title('Dataset2 - Perceptron')
    plt.show(block=True)

    # Adaline
    # Training
    neuron, errorEvolution = trainAdaline(df)

    # Plot error evolution
    plt.plot(errorEvolution)
    plt.xlabel('Iteration')
    plt.ylabel('Relative error')
    plt.title('Dataset2 - Adaline')
    plt.show(block=True)




if __name__ == '__main__':
    main()
