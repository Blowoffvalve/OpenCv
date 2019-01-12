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
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eq = cv2.equalizeHist(gray)
cv2.imshow("Original", gray)
cv2.imshow("eq", eq)
hStack = np.hstack([gray, eq])
cv2.imshow("Histogram Equalization", hStack)
cv2.waitKey()

#Save the histogram equalized comparison
r = 600.0/hStack.shape[1]
dim= (600, int(hStack.shape[0] * r))
eqCompare = cv2.resize(hStack, dim)
cv2.putText(eqCompare, "GrayScale", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
cv2.putText(eqCompare, "Histogram equalized", (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
cv2.imshow("StackedImageShrunk", eqCompare)
cv2.imwrite("images\StackedImageShrunk.jpg", eqCompare)
cv2.waitKey()

cv2.destroyAllWindows()