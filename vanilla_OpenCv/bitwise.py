import argparse
import cv2
import numpy as np
import imutils

"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
"""

#create a background and draw a rectangle on it
rectangle = np.zeros((300,300), dtype="uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)
cv2.waitKey()

#create a background and draw a Circle on it
circle = np.zeros((300,300), dtype="uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)
cv2.waitKey()

#bitwiseAnd
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("And", bitwiseAnd)
cv2.waitKey()

#bitwiseOr
bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("Or", bitwiseOr)
cv2.waitKey()

#bitwiseXor
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("Xor", bitwiseXor)
cv2.waitKey()

#bitwiseNot
bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("Not", bitwiseNot)
cv2.waitKey()

cv2.destroyAllWindows()