import numpy as np
import argparse
import cv2
import mahotas

args={}
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
"""
args["image"] = "images\coins.png"

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (11,11), 0)
cv2.imshow("Image", blur)
cv2.waitKey()

edges = cv2.Canny(blur, 60, 150)
cv2.imshow("Edges", edges)
cv2.waitKey()

(_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))

coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0,255,0), 2)
cv2.imshow("Coins", coins)
cv2.waitKey()

cv2.destroyAllWindows()

#To crop each coin from the image.
for (i,c) in enumerate(cnts):
    (x, y, w, h)= cv2.boundingRect(c)
    
    print("Coin #{}".format(i+1))
    coin = image[y:y+h, x:x+w]
    cv2.imshow("Coin", coin)
    
    mask = np.zeros(image.shape[:2], dtype="uint8")
    
    ((centerX, centerY), radius)= cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y+h, x:x+w]
    
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask=mask))
    cv2.waitKey(0)
