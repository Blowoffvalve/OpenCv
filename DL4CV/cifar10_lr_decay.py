#set matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

#import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from utilities.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

output = "output/lr_decay_f0.25_plot.png"
"""
import argparse
ap= argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
"""

def step_decay(epoch):
    #initialize the base learning rate, drop factor and the number of epochs after which the epoch should be dropped
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5
    
    #Calculate learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1+epoch)/dropEvery))
    
    #return the learning rate.
    return float(alpha)

#load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data")
((trainX, trainY), (testX, testY)) = cifar10.load_data(  )
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0
 
#convert the targets to vector/tensors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
 
#initialize the label names for the CIFAR-10 dataset
labelNames=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
 
#define the set of callbacks to be passed to the model during Learning
callbacks = [LearningRateScheduler(step_decay)]
 
#initialize the optimizer and model
opt = SGD(lr=0.01, momentum = 0.9, nesterov=True)
model=MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics = ["accuracy"])
 
#train the model
model.fit(trainX, trainY, validation_data = (testX, testY), batch_size=64, epochs=40, callbacks = callbacks)

#evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(trainX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(40), H.history["loss"], label="train_loss")
plt.plot(range(40), H.history["val_loss"], label="val_loss")
plt.plot(range(40), H.history["acc"], label="train_acc")
plt.plot(range(40), H.history["val_acc"], label="val_acc")
plt.title("Training loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(output)