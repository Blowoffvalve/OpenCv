#We build a classifier to predict some random data.
#import packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
#import argparse

def sigmoid_activation(x):
    #compute the sigmoid activation value for a given input
    return 1.0/(1+ np.exp(-x))

def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    #Apply the step function to threshold the output to binary class labels.
    preds[preds<=0.5]=0
    preds[preds>0] = 1
    
    #return the prediction
    return preds

epochs = 100
alpha = 0.01

#generate a 2-class classification problem with 1000 data points, where each is a 2D vector.
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

#Implement the bias trick by inserting a column of 1's as the first entry in the feature matrix
X = np.c_[np.ones(X.shape[0]), X]

#partition the data into training and test data
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size = 0.5, random_state = 42)

#Initialize our weight matrix and a list of losses so we can plot this over time.
print("[INFO] training....")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in range(epochs):
    preds = sigmoid_activation(trainX.dot(W))
    
    #We define our error as the difference between the predicted value and the actual label
    error = preds - trainY
    #Our loss is the sum of squared error.
    loss = np.sum(error**2)
    losses.append(loss)
    
    #The gradient is the dot product of X and the error
    gradient = trainX.T.dot(error)
    
    #The new weight can be gotten from the standard Weight update equation Wnew = Wold - (Lr* gradient) where Lr is learning rate. 
    #This is gradient descent
    W = W - (alpha * gradient)
    #print every 5 epochs
    if epoch ==0 or (epoch + 1)%5 ==0:
        print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch + 1), loss))
        
print("[INFO] evaluating")
preds = predict(testX, W)
print(classification_report(testY, preds))

#plot the testing classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 1], testX[:, 2], marker = "o",c = testY[:,0], s = 30)


#Plot the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()