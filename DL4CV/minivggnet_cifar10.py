#set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

#import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from utilities.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..\DL4CV")
output = "./output/cifar10_minivggnet_with_BN.jpg"
"""
import argparse
#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
"""
#load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

#convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#initalize the label names for he CIFAR-10 dataset
labelNames=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum = 0.9, nesterov=True)
model = MiniVGGNet.build(32, 32, 3, 10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=(["accuracy"]))

#train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, batch_size=64, epochs=40, validation_data=(testX, testY))

#evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names = labelNames))

model.save("./minivggnet-cifar10.hdf5")
#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(40), H.history["loss"], label = "train_loss")
plt.plot(range(40), H.history["val_loss"], label = "val_loss")
plt.plot(range(40), H.history["acc"], label= "train_acc")
plt.plot(range(40), H.history["val_acc"], label = "val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/ Accuracy")
plt.legend()
plt.savefig(output)