#import the necessary packages
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

dataset = "./datasets/animals"
modelDest = "shallownet_weights.hdf5"
"""
import argparse 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--modelDest", required = True, help="path to output model")
args = vars(ap.parse_args())
"""

#grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))

#initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

#load the dataset from disk then scale the raw pixel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, 500)
data = data.astype("float")/255.0

#Split into train test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size =0.2, random_state=42)

#Convert the targets to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#initialize the model and optimizer
print("[INFO] compiling the model...")
opt = SGD(0.005)
model = ShallowNet.build(32, 32, 3, 3)
model.compile(optimizer = opt, loss="categorical_crossentropy", metrics = ["accuracy"])

#train the network
print("[INFO] training the network")
H = model.fit(trainX, trainY, batch_size=32, epochs=100, verbose=1, validation_data=(testX, testY))

#Save the network to disk
print("[INFO] serializing the network...")
model.save(modelDest)

print("[INFO] evaluating network")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(100), H.history["loss"], label="train_loss")
plt.plot(range(100), H.history["val_loss"], label = "val_loss")
plt.plot(range(100), H.history["acc"], label = "train_acc")
plt.plot(range(100), H.history["val_acc"], label = "val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()