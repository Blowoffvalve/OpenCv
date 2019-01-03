from utilities.nn.perceptron import Perceptron
import numpy as np

#construct the OR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

#define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha = 0.1)
p.fit(X, y, epochs = 20)

#test our perceptron
print("[INFO] testing perceptron...")
for(x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data = {}, ground-truth = {}, pred = {}".format(x, target[0], pred))