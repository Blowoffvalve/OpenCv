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
cv2.waitKey()

(h,w) = image.shape[:2]
center = (w//2, h//2)

#rotate manually
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("Rotated by 45 degrees", rotated)
cv2.waitKey()

M=cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 90 degrees", rotated)
cv2.waitKey()

#call rotate from imutils 
rotated=imutils.rotate(image, 180)
cv2.imshow("Rotated by 180 degrees", rotated)
cv2.waitKey()

cv2.destroyAllWindows()