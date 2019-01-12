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

#flip Horizontally
flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)
cv2.waitKey()

#flip vertically
flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)
cv2.waitKey()

#flip horizontally and vertically
flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally and vertically", flipped)
cv2.waitKey()

cv2.destroyAllWindows()