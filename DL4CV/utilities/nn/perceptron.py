#import the necessary packages
import numpy as np

class Perceptron:
    def __init__(self, N, alpha = 0.1):
        #initialize the weight matrix(N+1 because we're using the bias trick and scale it to allow for faster convergence.
        #and store the learning rate
        self.W = np.random.randn(N+1)/np.sqrt(N)
        self.alpha= alpha
    
    #Step activation function.    
    def step(self, x):
        return 1 if x >0 else 0

    #Define the fit method to train the network
    def fit(self, X, y, epochs = 10):
        #Since i'm using bias trick, i insert 1's at the start of the X matrix
        X = np.c_[np.ones(X.shape[0]), X]
        for epoch in range(epochs):
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))
                
                if p != target:
                    error = p - target
                    #Update the weight
                    self.W = self.W - error * self.alpha * x
    
    #define the predict method to test the network
    def predict(self, X, addBias = True):
        #ensure our inupt is a matrix
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[np.ones(X.shape[0]), X]
        
        # take the dot product between the input features and the weight matrix then pass that through the step function.
        return self.step(np.dot(X, self.W))