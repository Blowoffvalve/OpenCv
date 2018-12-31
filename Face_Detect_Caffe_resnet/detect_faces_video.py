#import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pre-trained model" )
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#initialize the video stream and turn on the camera
print("Starting video stream...")
vs = VideoStream(src = 0).start()
#time.sleep cuz my camera acts wonky when i don't do this.
time.sleep(10.0)

#Loop over the frames from the video stream
while True:
    #Grab the fame from the stream and resize it to have a max width of 400
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w)= frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300,300), (104, 177, 123))
        
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            
            if confidence < args["confidence"]:
                continue
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            text = "{:.2f}%".format(confidence * 100)
            y = startY -10 if startY-10 > 10 else startY +10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        #If the q key was pressed, break
        if key == ord("q"):
            break
cv2.destroyAllWindows()
vs.stop()