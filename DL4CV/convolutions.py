#import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import cv2

image = "./datasets/MGA.jpg"
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parseargs())
"""
def convolve(image, K):
    #get the dimensions of the image and the kernel
    #Our objects have the shape(h, w, d)
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]
    
    #pad the image so the output image has the same dimensions as the input image.
    pad = (kW-1)//2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    #Create an output to be filled in
    output = np.zeros((iH, iW), dtype="float")
    
    #convolve the kernel over the image row-wise then height-wise. the y and x positions are centers that will get replaced.
    for y in range(pad, iH+pad):
        for x in range(pad, iW+pad):
            #extract the section of the image you're currently covolving over.
            roi = image[y-pad: y+pad+1, x-pad:x+pad+1]
            #perform the multiplication and sum operations
            out = (roi * K).sum()
            
            #Write the value you get for this roi to the respective location of the output
            output[y-pad, x-pad]=out
    #rescale the values in the output image, normalizing them and clipping values outside the range[0,255] to 0 and 1 respectively incase values outside that range occur
    output = rescale_intensity(output, in_range=(0,255))
    #Get back to a range [0, 255]
    output = (output * 255).astype("uint8")
    
    #return the output image.
    return output

#Constructing two average/blurring kernels to smooth an image
smallBlur = np.ones((7,7), dtype="float")*(1.0/(7*7))
largeBlur = np.ones((21,21), dtype = "float")*(1.0/(21*21))

#constructing a sharpening filter
sharpen = np.array(([0, -1, 0], 
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")

#construct the Laplacian kernel used to detect edge-like regions of an image.
laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype = "int")

#construct the sobel x-axis kernel
sobelX = np.array(([-1, 0, 1],
                   [1, -4,  1], 
                   [0, 1, 0]), dtype="int")

#construct the sobel y-axis kernel
sobelY = np.array(([-1, -2, -1], 
                   [0, 0, 0], 
                   [1, 2, 1]), dtype = "int")

#construct the emboss kernel
emboss = np.array(([-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]), dtype = "int")

#Construct a kernel bank, we can apply over the kernel
kernelBank = (
                ("small_blur", smallBlur),
                ("large_blur", largeBlur),
                ("sharpen", sharpen),
                ("laplacian", laplacian),
                ("sobel_x", sobelX),
                ("sobel_y", sobelY),
                ("emboss", emboss)
             )

#load the image and convert it to grayscale
image = cv2.imread(image)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, K) in kernelBank:
    print("[Info] applying {} kernel".format(kernelName))
    print(K.shape)
    convolveOutput = convolve(grayImage, K)
    opencvOutput = cv2.filter2D(grayImage, -1, K)
    
    #show the output images
    cv2.imshow("Original", grayImage)
    cv2.imshow("{} filter convolved".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()