import numpy as np
import argparse
import cv2
import mahotas

args={}
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
"""
args["image"] = "images\jp.png"

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5,5), 0)

cv2.imshow("original", image)
cv2.waitKey()

#Calculate the Otsu threshold for this image.
T = mahotas.thresholding.otsu(blurred)
print("Otsu's threshold: {}".format(T))

thresh = image.copy()
thresh[thresh>T]=255
thresh[thresh<255]=0
cv2.imshow("Otsu thresh", thresh)
cv2.waitKey()

#reverse the colors
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu thresh inversed", thresh)
cv2.waitKey()

#Calculate the riddle calvard threshold for this image
T = mahotas.thresholding.rc(blurred)
print("RC's threshold: {}".format(T))

thresh = image.copy()
thresh[thresh>T]=255
thresh[thresh<255]=0
cv2.imshow("RC thresh", thresh)
cv2.waitKey()

#reverse the colors
thresh = cv2.bitwise_not(thresh)
cv2.imshow("RC thresh inversed", thresh)
cv2.waitKey()