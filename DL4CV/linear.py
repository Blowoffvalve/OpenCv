#load relevant packages
import numpy as np
import cv2

#initialize the class labels and set the seed of the pseudorandom number generator so we can reproduce our results
labels = ["dog", "cat", "panda"]
np.random.seed(1)

#Randomly initialize our weights matrix and bias vector.
W = np.random.randn(3, 3072)
b = np.random.randn(3)

#load our example image and resize it, then flatten it to our feature vector representation
orig = cv2.imread("./datasets/beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

#compute the score by taking dot product of weight and images and adding the bias to it
scores = W.dot(image)+b

#loop over the scores and labels then display them
for(label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

#draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow("Image", orig)
cv2.waitKey(0)