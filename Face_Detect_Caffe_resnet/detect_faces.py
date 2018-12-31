import numpy as np
import argparse
import cv2

#Construct the argument parser and retrieve args from the command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to input image")
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe \
                'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe\
                pre-trained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = \
                "minimum probablility to filter weak detections")
args = vars(ap.parse_args())



#Load our serialized model from disk
print("[INFO] loading model....")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#Load the input image, resize to 300 by 300 and normalize 
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
print("The image has the shape, {}".format(image.shape))
#Preprocessing using blobfromimage to blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0,(300,300), (104.0, 177.0, 123.0))

#pass the blob through the network to get the detections
# and predictions
print("computing object detections...")
net.setInput(blob)
detections = net.forward()

#Loop over the detections. It is a 4 deep array with individual elements in the 3rd position and it's content within.
for i in range(0, detections.shape[2]):
    
    #get the confidence of each prediction. This is the 3 element of each detection.
    confidence = detections[0,0,i,2]
    
    #Check that the confidence is greater than the minimum confidence specified.
    
    if confidence > args["confidence"]:
       
        #the positions 3 to 6 are the location of the bounding box scaled over 100%. To get the actual position, just multiply by the width and height of the image.
        box = detections[0,0,0,3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype("int") 
        
        #Draw the bounding box of the face along with the associated probabililty
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY-10>10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),\
                      (0,0,255),2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX\
                    , 0.45, (0,0,225), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
print(detections)
print(detections.shape)