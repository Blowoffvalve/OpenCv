import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

args={}
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
"""
args["image"] = "images\jp.png"

image = cv2.imread(args["image"])

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
cv2.waitKey()
hist = cv2.calcHist([image], [0], None, [256], [0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.plot(hist)
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.xlim([0,256])
plt.show
