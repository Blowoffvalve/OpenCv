#import the necessary packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha = 0.1):
        #initialize the list of weights matrices, then store the network architecture and learning rates
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        #start looping from the index of the first layer but stop before we reach the last two layers
        for i in range(len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i+1]+1)
            #We normalize the values we are writing.
            self.W.append(w/np.sqrt(layers[i]))
        #The last two layers are special where the input connections need a bias term but the output doesn't
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))
        
    def __repr__(self):
        #construct and return a string that represennts the network
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))