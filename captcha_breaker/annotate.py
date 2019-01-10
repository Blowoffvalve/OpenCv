#Import the necessary packages
from imutils import paths
import argparse
import os
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True, help= "path to output directory of annotations")
args = vars(ap.parse_args())

#grab the image paths then intialize the dictionary of character counts
imagePaths = list(paths.list_images(args["input"]))
counts ={}

#Annotate the images using openCv
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    try:
       image = cv2.imread(imagePath) 
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       gray = cv2.copyMakeBorder(gray, 8,8,8,8, cv2.BORDER_REPLICATE)
       #threshold the image to convert to black and white only. THRESH_OTSU normalizes the thresholding an helps ensure your output contains the required details
       thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
       
       #find contours in the image keeping ony thr four largest ones
       cnts= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       cnts = cnts[0] if imutils.is_cv2() else cnts[1]
       cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
       #loop over the counters
       for c in cnts:
           #compute the bounding box for the contour then extract the digit.
           (x, y, w, h) = cv2.boundingRect(c)
           roi=gray[y-5:y+h+5, x-5:x+w+5]
           
           #display the character, making it large enough to see, then wait for a key
           cv2.imshow("ROI", imutils.resize(roi, width=28))
           key=cv2.waitKey(0)
           #If the "'" key is pressed, ignore the character
           if key == ord("'"):
               print("[INFO] ignoring character")
               continue
           #Use key press to make the output directory. 
           key = chr(key)
           if key.isalpha():
               key = key.upper()
           dirPath=os.path.sep.join([args["annot"], key])
           #if the output path doesn't exist, create it
           if not os.path.exists(dirPath):
               os.makedirs(dirPath)
           #write the labelled character to dirPath
           count = counts.get(key, 1)
           p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
           cv2.imwrite(p, roi)
            
           #increment the count for the current key
           counts[key] = count+1
    except KeyboardInterrupt:
       print("[INFO] manually leaving script")
       break
    #an unknown error has occurred for this particular image
    except:
        print("[INFO] skipping image...")

