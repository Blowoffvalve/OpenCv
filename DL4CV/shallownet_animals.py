#Packages packages packages
#import anti-gravity
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utilities.preprocessing.simple_preprocessor import SimplePreprocessor
from utilities.datasets.simple_dataset_loader import SimpleDatasetLoader
from utilities.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

dataset = "./datasets/animals"
#As usual, i comment the argument parser because it's easier from my IDE.
"""
#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "path to input dataset")
args = vars(ap.parse_args())
"""
imagepaths = list(paths.list_images(dataset))
#grab the list of images that we are working on
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

#Chain together the various preprocessors as we load images from the dataset.
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagepaths, verbose=500)
#scale the images to [0,1]
data = data.astype("float")/255.0

#partition the data into training and testing splits of 75:25
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

#intialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model= ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(optimizer = opt, loss="categorical_crossentropy", metrics=["accuracy"])

#train the network
print("[Info] training network")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

#Evaluate the network
print("[INFO] evaluating network")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(100), H.history["loss"], label = "train_loss")
plt.plot(range(100), H.history["val_loss"], label = "val_loss")
plt.plot(range(100), H.history["acc"], label = "train_acc")
plt.plot(range(100), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()