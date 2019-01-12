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
args["image"] = "../jp.png"

image = cv2.imread(args["image"])
cv2.imshow("original", image)
cv2.waitKey()

#crop Jessica's Face
cropped = image[10:220,80:290]
cv2.imshow("Crop of Jessica's Face", cropped)
cv2.waitKey()

cv2.destroyAllWindows()