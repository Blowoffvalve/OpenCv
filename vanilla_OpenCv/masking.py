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
cv2.imshow("original", image)
cv2.waitKey()

#crop Jessica's Face
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (80,10),(290,220), 255, -1)
cv2.imshow("mask", mask)
cv2.waitKey()
cv2.imwrite("images\MaskForJessicaFace.jpg", mask)

masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Masked", masked)
cv2.waitKey()
cv2.imwrite("images\MaskOfJessicaFace.jpg", masked)
cv2.destroyAllWindows()