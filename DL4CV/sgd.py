#import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def sigmoid_activation(x):
    return 1.0/(1 + np.exp(-x))

def predict(X, W):
    preds = X.dot(W)
    #Since we're still doing a binary classifier, use step function
    preds[preds <=0.5] = 0
    preds[preds  >0.5] = 1
    
    return preds

#Generator to get data from the X and Y elements
def next_batch(X, y, batchSize):
    for i in range(0, X.shape[0], batchSize):
        yield(X[i:i + batchSize], y[i:i+batchSize])
        
        
#Generate 2 cllass classification problem with 1000 data points where each data point is a 2D feature vector.
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

#Bias trick by adding a column of 1's in front of the X array
X = np.c_[np.ones(X.shape[0]), X]

#Split data into train test
(trainX, testX, trainY, testY) = train_test_split( X, y, test_size = 0.5, random_state=42)

print("[INFO] training...")
#Initialize our weight matrix and list of losses for future examination
W = np.random.randn(X.shape[1], 1)
losses = []

epochs = 100
batch_size = 32
alpha = 0.001

#learn
for epoch in range(epochs):
    epochLoss = []
    for(batchX, batchY) in next_batch(trainX, trainY, batch_size):
        #perform SGD for the batch
        preds = sigmoid_activation(batchX.dot(W))
        
        #the error
        errors = preds - batchY
        epochLoss.append(np.sum(errors**2))
        
        #the gradient descent step
        gradient = batchX.T.dot(errors)
        W = W - (alpha * gradient)
    #Calculate the loss over all the batches in an epoch.
    loss = np.average(epochLoss)
    losses.append(loss)
    
    #print every 5 epochs
    if epoch ==0 or (epoch+1)%5==0:
        print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch+1), loss))
    
#evaluate our model
print("[INFO] evaluating...")
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