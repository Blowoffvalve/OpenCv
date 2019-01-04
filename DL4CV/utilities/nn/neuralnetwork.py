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
    
    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1-x)
    
    def fit(self, X, y, epochs = 1000, displayUpdate = 100):
        #do the bias trick
        X = np.c_[np.ones(X.shape[0]), X]
        for epoch in range(epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
                
            if epoch ==0 or (epoch + 1)%displayUpdate==0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch = {}, loss = {:.7f}".format(epoch+1, loss))
                
    def fit_partial(self, x, y):
        #construct our list of output activations for each layer as our data point flows through the network; the first activation is a special case -- i
        A = [np.atleast_2d(x)]
        
        #Feedforward
        #loop over the layers in the network
        for layer in range(len(self.W)):
            net = A[layer].dot(self.W[layer])
            #compute the network output
            out = self.sigmoid(net)
            #Add the network's output to our list of activations
            A.append(out)
            
        #Backpropagation
        #The first phase of this is to compute the difference between our prediction(final output activation in the activations list ) and the correct value
        error = A[-1] - y
        
        #from here, we need to apply the chain rule and build our list of deltas 'D'; the first entry is the error of the output lyaer times the derivative of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]
        
        #traverse the activation
        for layer in range(len(A)-2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        D = D[::-1]
        
        #Weight update phase
        for layer in range(len(self.W)):
            self.W[layer] = self.W[layer] - self.alpha * A[layer].T.dot(D[layer])
            
    def predict(self, X, addBias = True):
        p = np.atleast_2d(X)
        if addBias:
            #Bias trick
            p = np.c_[np.ones(p.shape[0]), p]
        
        #loop over the layers in our network
        for layer in range(len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p
    
    def calculate_loss(self, X, targets):
        targets= np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) **2)
        
        #return the loss
        return loss