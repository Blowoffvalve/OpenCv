## Sample code from my goofing around with openCv.

### Experiments
1. Working with a single channel of an image might be sufficient for some tasks. It reduces the data that needs to go through your Conv by 2/3. 
2. Working with a different color space might also give you better classification accuracy.

### Heuristics
1. Arithmetic using the cv2.add and cv2.subtract methods ensure that the values stay in the range of [0 255] and do not wrap around that range.
2. When creating blank images(e.g. using np.zeros()) always ensure the dtype is "uint8" so the value is constrained to the range [0 255]. See Appendix 1.
3. We can apply masks using the bitwise operations(and, or, xor, not). See appendix 2.
4. Histogram equalization improves the contrast of an image by "stretching" the distribution of pixel intensity. See Appendix 3
5. The various blurs are used for different reasons
...A. To blend the images so each pixel is the average of all other pixels around it, use an **average blur**
            `cv2.blur(image, (3,3))`
...B. To blend the image with a configureable mean, use a **gaussian blur**. The output image tends to look more naturally blurred.
            `cv2.GaussianBlur(image, (3,3),0)`
...C. To reduce noise in an image, we can replace each pixel with the median of the pixels around it within a kernel using a **median blur**
            `cv2.medianBlur(image, 3)`
...D. To reduce noise while maintaining the edges in an image, use a **bilateral blur**
            `cv2.bilateralFilter(image, 5, 21, 21)`
<img src="images/BlurComparison.jpg">

### Appendix
1. Creating a black square background.

        **np.zeros((300,300), dtype="uint8")**

2. Using **bitwise and** to mask everything in the background as done with 

<img src="images/jpShrunk.png"> + <img src="images/RectangleMaskForJessicaFace.jpg"> = <img src="images/JessicaFaceRectangleMasked.jpg">   

        **OR**   

<img src="images/jpShrunk.png"> + <img src="images/CircleMaskForJessicaFace.jpg"> = <img src="images/JessicaFaceCircleMasked.jpg">.

3. Comparing a gray scale image to an image with the histogram equalized
        
<img src="images/StackedImageShrunk.jpg">