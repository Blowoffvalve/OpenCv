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

#Create a rectangular mask that captures Jessica's Face
rectangle = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(rectangle, (80,10),(290,220), 255, -1)
cv2.imshow("Mask", rectangle)
cv2.waitKey()

#Mask her face using a rectangle
rectangleMasked = cv2.bitwise_and(image, image, mask=rectangle)
cv2.imshow("Masked", rectangleMasked)
cv2.waitKey()

#create a circular mask that captures Jessica's ace
circle = np.zeros(image.shape[:2], dtype = "uint8")
(cX, cY) = (int((290-80)/2)+80, int((220-10)/2)+10)
cv2.circle(circle, (cX, cY), cY, 255, -1)
circleMasked = cv2.bitwise_and(image, image, mask=circle)
cv2.imshow("circleMasked", circleMasked)
cv2.waitKey()

#Write images to file
r = 200/image.shape[1]
dim = (200, int(image.shape[0] * r))

maskshrunk = cv2.resize(rectangle, dim)
cv2.imwrite("images\RectangleMaskForJessicaFace.jpg", maskshrunk)

imageShrunk = cv2.resize(image, dim)
cv2.imwrite("images\jpShrunk.png", imageShrunk)

output = cv2.resize(rectangleMasked, dim)
cv2.imwrite("images\JessicaFaceRectangleMasked.jpg", output)

maskshrunk = cv2.resize(circle, dim)
cv2.imwrite("images\CircleMaskForJessicaFace.jpg", maskshrunk)

output = cv2.resize(circleMasked, dim)
cv2.imwrite("images\JessicaFaceCircleMasked.jpg", output)

cv2.destroyAllWindows()