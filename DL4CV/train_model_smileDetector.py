from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from utilities.nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

args = {}
"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())
"""
args["dataset"] ="datasets/SMILEsmileD"
args["model"] = "weights/lenetSmile.hdf5"

#initialize the list of data and labels
data = []
labels = []

#loop over the input images
for imagePath in (paths.list_images(args["dataset"])):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)
    label=imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)
    
#scale the raw pixel to range [0 1]
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

#COnvert the labels to vectors
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)
    
#account for skew in balance by computing class weights
classTotals = labels.sum(axis=0)
classWeight = classTotals.max()/classTotals

#partition the data into 80%train, 20%test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=.2, stratify=labels, random_state=42)

#initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["accuracy"])

#train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight, batch_size=64, epochs=15)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=None)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] Serializing network...")
model.save(args["model"])

#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(15), H.history["loss"], label="train_loss")
plt.plot(range(15), H.history["val_loss"], label = "val_loss")
plt.plot(range(15), H.history["acc"], label = "train_acc")
plt.plot(range(15), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()