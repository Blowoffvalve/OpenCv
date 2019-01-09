#import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from utilities.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import cifar10
import os

weights = "weights/improvements"
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_arguments("-w", "--weights", required=True, help="path to weights directory")
args = vars(ap.parse_args())
"""
#Load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

#convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

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

#construct the callback to save only models that have lowered the validation loss to disk.
fname = os.path.sep.join([weights, "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode = "min", save_best_only=True, verbose=1)

callbacks = [checkpoint]
#train the network
print("[INFO] training network...")
H= model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)