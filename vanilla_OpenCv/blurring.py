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

#Average blur
blurred = np.hstack([
                        cv2.blur(image, (3,3)),
                        cv2.blur(image, (5,5)),
                        cv2.blur(image, (7,7))
                    ])

cv2.imshow("Averaged", blurred)
cv2.waitKey()

#Gaussian blur
blurred = np.hstack([
                    cv2.GaussianBlur(image, (3,3), 0),
                    cv2.GaussianBlur(image, (5,5), 0),
                    cv2.GaussianBlur(image, (7,7), 0)
                    ])
cv2.imshow("Gaussian", blurred)
cv2.waitKey()

#Median Blur
blurred = np.hstack([
                    cv2.medianBlur(image, 3),
                    cv2.medianBlur(image, 5),
                    cv2.medianBlur(image, 7)
                    ])

cv2.imshow("Median", blurred)
cv2.waitKey()

#Bilateral blurring
blurred = np.hstack([
                    cv2.bilateralFilter(image, 5, 21, 21),
                    cv2.bilateralFilter(image, 7, 31, 31),
                    cv2.bilateralFilter(image, 9, 41, 41)])
cv2.imshow("Bilateral", blurred)
cv2.waitKey()

#Write the various blurs to a file
r = 300/ image.shape[1]
dim = (300, int(image.shape[0] * r))

#average Blur 5
averageBlur5 = cv2.blur(image, (5,5))
cv2.putText(averageBlur5, "Average Blur 5*5 kernel", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2)
averageBlur5 = cv2.resize(averageBlur5, dim)

#gaussian Blur 5
gaussianBlur5 = cv2.GaussianBlur(image, (5,5), 0)
cv2.putText(gaussianBlur5, "Gaussian Blur 5*5 kernel", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2)
gaussianBlur5 = cv2.resize(gaussianBlur5, dim)

#median Blur 5
medianBlur5 = cv2.medianBlur(image, 5)
cv2.putText(medianBlur5, "Median Blur 5*5 kernel", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2)
medianBlur5 = cv2.resize(medianBlur5, dim)

#Bilateral Filter 5
bilateralFilter5 = cv2.bilateralFilter(image, 5, 21, 21)
cv2.putText(bilateralFilter5, "Bilateral Filter 5*5 kernel", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2)
bilateralFilter5 = cv2.resize(bilateralFilter5, dim)

outputs = np.vstack([
                    np.hstack([averageBlur5, gaussianBlur5]),
                    np.hstack([medianBlur5, bilateralFilter5])])
cv2.imshow("All 4 blurs", outputs)
cv2.waitKey()
cv2.imwrite("images/BlurComparison.jpg", outputs)
cv2.destroyAllWindows()