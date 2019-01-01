# -*- coding: utf-8 -*-
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths

dataset = "./datasets/animals"
imagePaths = list(paths.list_images(dataset))

#Initialize the image preprocessor, load the dataset from disk and flatten the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

#Encode the labels as integers
le = LabelEncoder()
labels = le.fit_transforms(labels)

#split the data into train and test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 5)

#loop over our regularizers
for r in (None, "l1", "l2"):
    #train a SGD classifier using a softmax loss function and the specified regularization function for 10 epochs
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(loss = "log", penalty=r, max_iter=10, learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)
    
    #evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r, acc * 100))