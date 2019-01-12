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

def plot_histogram(image, title, mask=None):
    chans = cv2.split(image)
    colors = ["B", "G", "R"]
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
               
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0,256])
        plt.plot(hist, color=color)
        plt.xlim([0,256])
        
image = cv2.imread(args["image"])        
cv2.imshow("Original", image)
cv2.waitKey()
plot_histogram(image, "Histogram for original image")

mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask, (80,10),(290,220), 255, -1)
cv2.imshow("mask", mask)
cv2.waitKey()

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("masked", masked)
cv2.waitKey()

plot_histogram(image, title="Histogram for masked Image", mask=mask)
cv2.destroyAllWindows()