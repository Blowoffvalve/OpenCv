import argparse
import cv2
import numpy as np
import imutils

args={}
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
"""
args["image"] = "..\jp.png"

image = cv2.imread(args["image"])
cv2.imshow("original", image)
cv2.waitKey()

#resize manually long width maintaining aspect-ratio using dim
r= 150.0/image.shape[1]
dim = (150, int(image.shape[0]*r))
resized = cv2.resize(image, dim)
cv2.imshow("Rotated by 180 degrees", resized)
cv2.waitKey()

#resize manually along height maintaining aspect-ratio
r = 50.0/image.shape[0]
dim = (int(image.shape[1]*r), 50)
resized =  cv2.resize(image, dim)
cv2.imshow("Rotated by 180 degrees", resized)
cv2.waitKey()

#call resize from imutils 
rotated=imutils.resize(image, height=200)
cv2.imshow("Rotated by 180 degrees", rotated)
cv2.waitKey()


cv2.destroyAllWindows()