#import the necessary packages
import sys
sys.path.append("..\DL4CV")
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from utilities.nn.conv.lenet import LeNet
from utilities.captchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

args= {}
args["dataset"]= "dataset"
args["output"] = "output/lenet5"
#Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

#initialize the data and labels
data=[]
labels=[]
#loop over the input images
for imagePath in paths.list_images(args["dataset"]):
    #load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)
    
    #get the class
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
#scale the pixel intensity to the range [0 1]
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

#partition the data into 75%training and 25%testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#convert the labels into vectors
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

#initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=9)
opt=SGD(lr=0.01)
model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics=["accuracy"])
#train the network
print("[INFO] training the network...")
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size=32, epochs=15, verbose=1)

#evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names = lb.classes_))

#save the model to disk
print("[INFO] serializing model")
model.save(args["model"])

#plot the losses and accuracies
#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(15), H.history["loss"], label = "train_loss")
plt.plot(range(15), H.history["val_loss"], label = "val_loss")
plt.plot(range(15), H.history["acc"], label= "train_acc")
plt.plot(range(15), H.history["val_acc"], label = "val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/ Accuracy")
plt.legend()
plt.show