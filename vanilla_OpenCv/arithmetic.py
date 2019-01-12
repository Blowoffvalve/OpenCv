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

print("max of 255: {}".format(cv2.add(np.uint8([200]), np.uint8([100]))))
print("min of 0: {}".format(cv2.subtract(np.uint8([50]), np.uint8([100]))))

print("wrap around addition {}".format(np.add(np.uint8([100]), np.uint8([200]))))
print("wrap around subtrsaction {}".format(np.subtract(np.uint8([100]), np.uint8([200]))))

M = np.ones(image.shape, dtype="uint8")*100
added = cv2.add(image, M)
cv2.imshow("Added", added)
cv2.waitKey()

M = np.ones(image.shape, dtype="uint8")*50
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey()

cv2.destroyAllWindows()