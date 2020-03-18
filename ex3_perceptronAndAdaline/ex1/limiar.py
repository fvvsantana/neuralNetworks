import pandas as pd
import random as rd
import inc.activationFunctions as af
from inc.perceptron import Perceptron
from inc.adaline import Adaline

def trainPerceptron(df, minLearningRate = 0.001, maxIter = 1000):
    neuron = Perceptron()
    acc = 0
    iter = 0
    learningRate = 1
    lastAccuracies = [0,1,0,1,0,1]

    # Track error during training
    errorEvolution = list()

    # Generate random weights and biases:
    range = 5
    neuron.weights = [rd.uniform(-range, range), rd.uniform(-range,range)]
                    #rd.uniform(-range, range), rd.uniform(-range,range)]
    neuron.bias = rd.uniform(-range, range)

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

    return {'neuron': neuron, 'errorEvolution': errorEvolution}

# Loop over dataset once, update weights and return accuracy
def iteratePerceptronTraining(df, neuron, weightUpdateStep = 1):
    hits = 0
    for index, row in df.iterrows():
        # Classify instance
        inputs = (row.V1, row.V2)
        neuronOutput = neuron.run(inputs)
        if neuronOutput > 0.5:
            classification = 2
        else:
            classification = 1

        # Update weights
        trueClass = row.V3
        for i in range(len(neuron.weights)):
            neuron.weights[i] += weightUpdateStep * inputs[i] * (trueClass - (neuronOutput + 1))

        # Update hits
        if trueClass == classification:
            hits += 1

    return hits/len(df)

def main():
    # Read csv file to dataframe
    df = pd.read_csv('../data/Aula3-dataset_1.csv')

    # Dataset 1

    # Perceptron
    # Training
    results = trainPerceptron(df)
    print('Weights:', results['neuron'].weights)
    print('Bias:', results['neuron'].bias)
    print('Iterations:', len(results['errorEvolution']))
    print('Error evolution:', results['errorEvolution'])


    # Plot error evolution

    # Adaline
    # Training

    # Plot error evolution

    # Dataset 2

    # Perceptron
    # Training

    # Plot error evolution

    # Adaline
    # Training

    # Plot error evolution


if __name__ == '__main__':
    main()
