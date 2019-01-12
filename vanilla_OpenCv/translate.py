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
args["image"] = "jp.png"

image = cv2.imread(args["image"])
cv2.imshow("original", image)

#Doing it manually
M = np.float32([[1,0,25], 
                [0,1,50]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Shifted down and right", shifted)
cv2.waitKey()

M = np.float32([[1,0,-50],
               [0,1,-25]])
shifted = cv2.warpAffine( image, M,(image.shape[1], image.shape[0]))
cv2.imshow("Shifted up and left", shifted)
cv2.waitKey()

#Using an imutils i wrote.
shifted = imutils.translate(image, 0, 100)
cv2.imshow("Shifted down", shifted)
cv2.waitKey()

