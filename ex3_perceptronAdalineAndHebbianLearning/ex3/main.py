'''

I used a package called seaborn that you need to install to run this code.
You can install it this way:
    pip install --user seaborn

'''
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score


import inc.activationFunctions as af
from inc.perceptron import Perceptron
from inc.adaline import Adaline

'''
Do the hebbian training until stop learning.
Here, clustering level is defined using the metric silhouette
df is the dataset without the label column.
'''
def hebbianTraining(df, minLearningRate = 0.001, maxIter = 100):
    neuron = Adaline()
    acc = 0
    iter = 0
    learningRate = 1
    lastClusteringLevel = list()
    for i in range(10):
        lastClusteringLevel.append(-1)
        lastClusteringLevel.append(1)

    # Track clustering level during training
    clusteringEvolution = list()

    # Generate random weights and biases:
    '''
    radius = 0.5
    neuron.weights = list()
    for i in range(df.shape[1]):
        neuron.weights.append(rd.uniform(-radius, radius))
    neuron.bias = rd.uniform(-radius, radius)
    '''
    neuron.weights = [-0.5, 1]
    neuron.bias = 0

    '''
    # Initialize weight and bias
    neuron.weights = np.zeros(df.shape[1])
    neuron.bias = 0
    '''

    averageByFeature = df.mean().values.tolist()
    print('ABF:', averageByFeature)

    while (learningRate > minLearningRate) and (iter < maxIter):
        # Loop over dataset once, update weights and return accuracy
        clusteringLevel = iterateHebbianTraining(df, neuron, averageByFeature)

        # Update learningRate
        del lastClusteringLevel[0]
        lastClusteringLevel.append(clusteringLevel)
        learningRate = max(lastClusteringLevel) - min(lastClusteringLevel)

        # Update error
        clusteringEvolution.append(clusteringLevel)
        # Update iteration counter
        iter += 1


    return neuron, clusteringEvolution

# Loop over dataset once, update weights and return silhouette
def iterateHebbianTraining(df, neuron, averageByFeature, weightUpdateStep = 1):
    # List of classifications during training to calculate silhouette
    classifications = list()

    has1 = False
    has2 = False
    # Average of neuron output to calculate the weights and bias updates
    neuronOutputAverage = 0
    for index, row in df.iterrows():
        # Classify instance
        inputs = row.values.tolist()

        neuronOutput = neuron.run(inputs)


        # Update neuronOutput average considering only the neuronOutputs
        # triggered up to now
        neuronOutputAverage = (index * neuronOutputAverage + neuronOutput) / (index + 1)
        print('nOA:', neuronOutputAverage)

        # Update weights
        for i in range(len(neuron.weights)):
            neuron.weights[i] += weightUpdateStep * (inputs[i] - averageByFeature[i]) * (neuronOutput - neuronOutputAverage)
            print('Delta pos', i, ':', weightUpdateStep * (inputs[i] - averageByFeature[i]) * (neuronOutput - neuronOutputAverage))
        neuron.bias -= weightUpdateStep * (neuronOutput - neuronOutputAverage)

        print(neuron.weights)
        print(neuron.bias)

        # Save classification
        if neuronOutput > 0.5:
            classifications.append(2)
            has2 = True
        else:
            classifications.append(1)
            has1 = True


    if has1 and has2:
        return silhouette_score(df, classifications)
    else:
        return -1

def main():
    # Dataset 1
    # Load and show dataset
    df = pd.read_csv('../data/Aula3-dataset_1.csv')
    #df = pd.read_csv('../data/test.csv')
    #df = df.sample(frac=1).reset_index(drop=True)
    #print(silhouette_score(df.iloc[:, :-1], df.iloc[:, -1]))
    #exit()


    sns.pairplot(df, hue=df.columns[-1])
    plt.title('Dataset1 - True')
    plt.show(block=True)

    # Training
    neuron, clusteringEvolution = hebbianTraining(df.iloc[:, :-1])
    print(clusteringEvolution)

    # Plot clustering evolution
    plt.plot(clusteringEvolution)
    plt.title('Dataset1 - Hebbian Learning')
    plt.xlabel('Iteration')
    plt.ylabel('Clustering evolution')
    plt.show(block=True)


if __name__ == '__main__':
    main()
