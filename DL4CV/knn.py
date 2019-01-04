#import libraries

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

#Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help ="path to input dataset")
ap.add_argument("-k", "--neighbors", type = int, default =1, help = "# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type= int, default = 1, help = "# of jobs for k-NN distance(-1 uses all available cores)")
args = vars(ap.parse_args())

#grab the list of images we'll be describing
print("[INFO] loading images....")
imagePaths = list(paths.list_images(args["dataset"]))

#initialize the image preprocessor, load the data from disk and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(imagePaths, verbose = 500)
print("[INFO] the current shape of the images object is {}".format(data.shape))
#Flatten each image into one contiguous vector.
data= data.reshape((data.shape[0], 3072))

#Show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB". format(data.nbytes/(1024*1000.0)))

#encode the labels as integers
le= LabelEncoder()
labels = le.fit_transform(labels)

#partition the data into training and testing splits with 75%-25% train-test split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors = args["neighbors"], n_jobs = args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names= le.classes_))
