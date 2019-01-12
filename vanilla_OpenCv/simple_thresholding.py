import numpy as np
import argparse
import cv2

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

#Do a binary threshold
(T, thresh) = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

(T, threshInv) = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)

cv2.imshow("People", cv2.bitwise_and(image, image, mask=thresh))
cv2.waitKey()
cv2.destroyAllWindows()
