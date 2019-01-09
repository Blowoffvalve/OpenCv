#import the necessary packages
from keras.applications import VGG16, VGG19, Xception, InceptionV3, ResNet50, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import cv2

imagePath = ".\datasets\example_images\example_05.jpg"
model = "resnet"
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i",  "--imagePath", required=True, help="path to the input image")
ap.add_argument("-model", "--model", required=True, help="name of pre-trained network to use")
args = vars(ap.parse_args())
"""

MODELS = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "inception": InceptionV3,
            "xception" : Xception,
            "resnet": ResNet50
          }

#ensure a valid model name was supplied via command line argument
if model not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in the Models dictionary")

#initialize the input image shape (224*224) along with the pre-processing function(this might need to be changed based on the model we use)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

#Inception and Xception have input shape of (299*299) and use a different image processing function
if model in("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input

#load the network weights from disk
print("[INFO] loading {}...".format(model))
Network = MODELS[model]
model = Network(weights="imagenet")

print("[INFO] loading and pre-processing image...")
image = load_img(imagePath, target_size=inputShape)
image = img_to_array(image)

#add a dimension to the image as required by the network to take it from [width, height, depth] to [index, width, height, depth] as required  by the network.
image = np.expand_dims(image, axis = 0)

#pre-process the image 
image = preprocess(image)

#classify the image
print("[INFO] classifying image with '{}'...".format(model))
preds= model.predict(image)
P = imagenet_utils.decode_predictions(preds)

#loop over the predictions and display the rank-5 prediction + probablities to the terminal
for(i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i+1, label, prob*100))
    
#load the image and draw the top prediction on the image and display the image on our screen
orig = cv2.imread(imagePath)
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)