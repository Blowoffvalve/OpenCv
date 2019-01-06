#import the necessary packages
from utilities.nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#load the dataset from disk.
print("[INFO] loading dataset...")
filename="./datasets/mldata/mnist-original.mat"
dataset = scipy.io.loadmat(filename)
data = dataset["data"]

#Reshape the dataset to math the ordering
if K.image_data_format()== "channels_first":
    data = data.reshape(data.shape[-1], 1, 28, 28)
else:
    data = data.reshape(data.shape[-1], 28, 28, 1)

#scale the input data to the range [0,1] and split the data into train/test
data = data.astype("float")/255.0
target = dataset["label"].astype("int")
target = target.reshape((target.shape[1], target.shape[0]))
target.astype("int")
(trainX, testX, trainY, testY) = train_test_split(data, target, test_size=0.25, random_state=42)

#convert the labels to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#initialize optimizer and mode
print("[INFO] configuring model")
optim = SGD(lr=0.01)
model = LeNet.build(28, 28, 1, 10)
model.compile(loss = "categorical_crossentropy", metrics = ["accuracy"], optimizer=optim)

#train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)

#evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

#plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(20), H.history["loss"], label = "train_loss")
plt.plot(range(20), H.history["val_loss"], label = "val_loss")
plt.plot(range(20), H.history["acc"], label= "train_acc")
plt.plot(range(20), H.history["val_acc"], label = "val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/ Accuracy")
plt.legend()
plt.show()