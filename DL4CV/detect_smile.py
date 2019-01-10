#import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
"""
#construct the argumetn parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help = "path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained sile detector CNN")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())
"""
args={}
args["cascade"] = ".\haarcascade_frontalface_default.xml"
args["model"] = ".\weights\lenetSmile.hdf5"
args["video"]

#load the face detector cascade
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

#if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])
    
while True:
    #grab the current frame
    (grabbed, frame) = camera.read()
    
    #I we are viewing a video and we did not grab a frame, then we have reach the end of the video
    if args.get("video") and not grabbed:
        break
    
    #resize the frame, convert to grayscale and cole the frame to draw on it later in the program
    frame= imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    
    #detect faces in the input frame, then clone the frame so we can draw on it.
    rects= detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    for (fX, fY, fW, fH) in rects:
        #extract the ROI of the face from the grayscale image, resize it into a fixed 28X 28 picel and prepare the ROI for classification via the CNN.
        roi = gray[fY:fY+fH, fX:fX+fW]
        roi = cv2.resize(roi, (28,28))
        roi = roi.astype("float")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        #determine the probabilities of both smiling and not smiling then set the label accordingly
        (notSmiling, smiling) = model.predict(roi)[0]
        label="smiling" if smiling > notSmiling else "Not Smiling"
        #display the label and bounding box rectangle on the output frame
        cv2.putText(frameClone, label, (fX, fY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0,0,255), 2)
        
        #showour detected faces along with the output label to our screen:
        cv2.imshow("Face", frameClone)
        if cv2.waitKey(1) and 0xFF == ord("q"):
            break
#cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()