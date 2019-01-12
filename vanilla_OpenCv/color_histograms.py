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
cv2.imshow("Original", image)

chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
           
#Splitting into channels and plotting the histogram per channel
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0,256])
    plt.plot(hist, color=color)
    plt.xlim([0,256])

#plotting 2D histograms
fig = plt.figure()
#Plot histogram for G and B
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0,256,0,256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histograms for G and B")
plt.colorbar(p)

#Plot histogram for G and R
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0,256,0,256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histograms for G and R")
plt.colorbar(p)


#Plot histogram for B and R
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0,256,0,256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histograms for B and R")
plt.colorbar(p)
plt.savefig("multiColor Hist.jpg")
print("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))


#Plotting 3D histograms
hist= cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
print("3D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))
plt.show()