import pandas as pd
import numpy as np
import math
import random

"""
MLP Algorithm from scratch by Benjamin Hake
Steps:
1. Data
    a. import data using pandas
    b. Normalize data to values between 0 - 1
    c. Seperate Data using 5 fold cross validation
2. MLP Algorithm
    1. Training
        a. Initialize weights using range reccomended by slides
        b. feed forward training in range of max iterations using sigmoid function as activation function for both layers
        c. back propogation after each input using derivative of sigmoid functions
    2. Testing
        a. Iterate through testing inputs
        b. Round activation to nearest output
        c. Divide amount of correctly guessed records by amount of training data to retrieve accuracy
        d. compare MLP guess accuracy to random guess accuracy
3. Runner
    a. run algorithm through 5 training and testing sets provided by Data
    b. Report Accuracy with each training and testing set
    c. compare mean accuracy of mlp algorithm and random guess algorithm
"""
class Data:
    def __init__(self, file):
        self.data = pd.read_csv(file, header=None)
        self.dimensions = self.data.shape[1] - 1
        self.normalizeData()
        self.trainingDatas = []
        self.testingDatas = []
        self.seperateData()
        
    def normalizeData(self):
        # Normalizes features to between 0 and 1 using (x - xmin) / (xmax - xmin)
        for i in range(1, self.dimensions+1):
            finarr = []
            dimarr = self.data[i].to_numpy()
            mini = np.amin(dimarr)
            maxi = np.amax(dimarr)
            
            for j in range(len(self.data)):
                finarr.append((dimarr[j] - mini) / (maxi - mini))
            self.data[i] = finarr

    def seperateData(self):
        # shuffles training data
        randomlyIteratedData = self.data.sample(frac=1)
        # splits shuffled data into 5 
        self.testingDatas = np.array_split(randomlyIteratedData, 5)
        for i in self.testingDatas:
            # Takes differences of Data and testing sets respectively, assumes no duplicates in data
            self.trainingDatas.append(pd.concat([randomlyIteratedData, i]).drop_duplicates(keep=False))
    

class HiddenNeuron:
    def __init__(self, dimensions):
        # Initialize weights using algorithm from slides
        self.weights = np.random.uniform(low = -(1/math.sqrt(dimensions)), high = 1/math.sqrt(dimensions), size = dimensions)
        
class HiddenLayer:
    def __init__(self, amountOfNeurons, dimensions):
        self.dimensions = dimensions
        self.bias = 0.0
        self.neurons = []
        # Initialize Hidden Layer Neurons
        for i in range(amountOfNeurons):
            self.neurons.append(HiddenNeuron(self.dimensions))
    
    # Puts weights Into Single Array
    def getWeightsArray(self):
        finarr = []
        for neuron in self.neurons:
            finarr.append(neuron.weights)
        return finarr
    
    def updateWeights(self, arr):
        for i, neuron in enumerate(self.neurons):
            neuron.weights = arr[i]

class OutputNeuron:
    def __init__(self, amountOfHiddenNeurons):
        # Initialize weights with algorithm from slides
        self.weights = np.random.uniform(low = -(1/math.sqrt(amountOfHiddenNeurons)), high = 1/math.sqrt(amountOfHiddenNeurons), size = amountOfHiddenNeurons)


class OutputLayer:
    def __init__(self, amountOfHiddenNeurons):
        self.bias = 0.0
        # Initialize Output Layer neurons
        self.neuron = OutputNeuron(amountOfHiddenNeurons)
    
    def getWeightsArray(self):
        return self.neuron.weights
    # Update weights from back propogation
    def updateWeights(self, arr):
        self.neuron.weights += arr
    
class MultiLayerPerceptron:
    def __init__(self, trainingData, testingData, dimensions):
        # Initialize training data, training targets, testing targets, expected outputs of activation functions
        self.trainingDataUncut = trainingData
        self.testingData = testingData
        self.dimensions = dimensions
        self.trainingTargets = self.trainingDataUncut[0].to_numpy()
        self.testingTargets = self.testingData[0].to_numpy()
        self.testingData = testingData.drop(0, axis = 1, inplace = False).to_numpy()
        self.trainingTargetsOutput = []
        for i in self.trainingTargets:
            if i == 1:
                self.trainingTargetsOutput.append(0.0)
            elif i == 2:
                self.trainingTargetsOutput.append(.5)
            elif i == 3:
                self.trainingTargetsOutput.append(1.0)
        self.trainingData = self.trainingDataUncut.drop(0, axis = 1, inplace = False)
        self.trainingData = self.trainingData.to_numpy()
        
        # Initialize hidden layer using algorithm found online for amount of Hidden Layer Neurons
        self.amountOfHiddenNeurons = math.ceil(self.dimensions*(2/3))+1
        self.hiddenLayer = HiddenLayer(self.amountOfHiddenNeurons, self.dimensions)

        # Initialize output layer
        self.outputLayer = OutputLayer(self.amountOfHiddenNeurons)

        # Initialize Learning Rate and amount of iterations
        self.learningRate = 1
        self.maxIterations = 600
        # Accuracies to be used in testing
        self.accuracy = 0
        self.randomGuessAccuracy = 0
        
        # Run training
        self.feedForwardTraining()
        # Run Testing
        self.feedForwardTesting()
    
    # Used to Shuffle the training data after each iteration, as to not train over same permutation of data each iteration
    def shuffleData(self):
        self.trainingDataUncut = self.trainingDataUncut.sample(frac = 1)
        self.trainingTargets = self.trainingDataUncut[0].to_numpy()
        self.trainingData = self.trainingDataUncut.drop(0, axis = 1, inplace = False).to_numpy()
        self.trainingTargetsOutput = []
        for i in self.trainingTargets:
            if i == 1:
                self.trainingTargetsOutput.append(0.0)
            elif i == 2:
                self.trainingTargetsOutput.append(0.5)
            elif i == 3:
                self.trainingTargetsOutput.append(1.0)
        self.trainingTargetsOutput = np.asarray(self.trainingTargetsOutput)
        
    def feedForwardTesting(self):
        correct = 0
        randomGuessCorrect = 0
        for i, x in enumerate(self.testingData):
            realoutput = self.testingTargets[i]
            hiddenLayer = []
            for neuron in self.hiddenLayer.neurons:
                hiddenLayer.append(self.sigmoid(np.dot(neuron.weights, x)))
            outputLayer = self.sigmoid(np.dot(self.outputLayer.neuron.weights, hiddenLayer))
            # Round output layer activation function to nearest output, then retrieve MLP guess
            if outputLayer <= (1/3):
                output = 1
            elif outputLayer > 1/3 and outputLayer <= 2/3:
                output = 2
            else:
                output = 3
            # MLP guess correctly
            if output == realoutput:
                correct += 1
            if self.randomGuess() == realoutput:
                randomGuessCorrect += 1
        n = len(self.testingTargets)
        # Compute Final testing accuracy
        self.accuracy = correct / n
        self.randomGuessAccuracy = randomGuessCorrect / n
    
    def backPropogation(self, inputs, hidden, output, expectedOutput):
        # Derivative Function output layer
        deltaO = (expectedOutput - output)*output*(1.0-output)
        # Get Hidden Weights(w1) and Output Weights(w2)
        w1 = self.hiddenLayer.getWeightsArray()
        w2 = self.outputLayer.getWeightsArray()
        # Derivative Function Hidden Layer with respect to output layer derivative function
        deltaH = hidden * (1.0 - hidden) * (np.dot(deltaO, np.transpose(w2)))
        # Initialize weight update arrays 
        w1add = np.zeros(np.shape(w1))
        w2add = np.zeros(np.shape(w2))
        # calculate weight updates my dot product of previous layer and relative derivative function
        w1add = self.learningRate * np.dot(np.transpose([inputs]), [deltaH])
        w2add = self.learningRate * np.dot(np.transpose(hidden), deltaO)
        w1 += np.transpose(w1add)
        # Update weights
        self.hiddenLayer.updateWeights(w1)
        self.outputLayer.updateWeights(w2add)
    
    def feedForwardTraining(self):
        # Number of times ran through training data set
        for i in range(self.maxIterations):
            # Shuffle Data after each generation
            self.shuffleData()
            hiddenLayerCache = []
            outputLayerCache = []
            # Enumerate so that we can retrieve expected output from TrainingTargetsOutput array
            for j, x in enumerate(self.trainingData):
                # Initialize Array to hold hidden layer activations
                hiddenLayer = []
                # Loop through hidden neurons
                for neuron in self.hiddenLayer.neurons:
                    # Activation Function
                    hiddenLayer.append(self.sigmoid(np.dot(neuron.weights, x) + self.hiddenLayer.bias))
                hiddenLayer = np.asarray(hiddenLayer)
                hiddenLayerCache.append(hiddenLayer)
                # Retrieve Output Neuron
                outneur = self.outputLayer.neuron
                # Output layer activation function
                output = self.sigmoid(np.dot(hiddenLayer, outneur.weights) + self.outputLayer.bias)
                outputLayerCache.append(output)
                # Run back propogation using inputs, hidden layer activations, output layer activations, expected output
                self.backPropogation(x, hiddenLayer, output, self.trainingTargetsOutput[j])

            
    def randomGuess(self):
        return random.randint(1, 4)

    def sigmoid(self, h):
        return 1.0 / (1.0 + math.exp(-h))
    
    def tanh(self, h):
        return (math.exp(h)-math.exp(-h))/(math.exp(h)+math.exp(-h))

class Runner:
    def __init__(self, data):
        mlpaccuracies = np.zeros(5)
        randomguessaccuracies = np.zeros(5)
        for i in range(5):
            print(" ")
            mlp = MultiLayerPerceptron(data.trainingDatas[i], data.testingDatas[i], data.dimensions)
            mlpaccuracies[i] = mlp.accuracy
            randomguessaccuracies[i] = mlp.randomGuessAccuracy
            print("MLP accuracy ", i+1, ": ", mlp.accuracy)
            print("Random guess accuracy ", i+1, ": ", mlp.randomGuessAccuracy)
        print(" ")
        print("mean MLP accuracy: ", np.mean(mlpaccuracies)) 
        print("mean Random Guess accuracy: ", np.mean(randomguessaccuracies))
   
r = Runner(Data('wine.data'))
