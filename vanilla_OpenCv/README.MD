## Sample code from my goofing around with openCv.



### Heuristics
1. Arithmetic using the cv2.add and cv2.subtract methods ensure that the values stay in the range of [0 255] and do not wrap around that range.
2. When creating blank images(e.g. using np.zeros()) always ensure the dtype is "uint8" so the value is constrained to the range [0 255]. See Appendix 1.
3. We can apply masks using the bitwise operations(and, or, xor, not). See appendix 2.

### Appendix
1. Creating a black square background.
        **np.zeros((300,300), dtype="uint8")**

2. Using **bitwise and** to mask everything in the background as done with 

        <img src="images/jpShrunk.png"> + <img src="images/RectangleMaskForJessicaFace.jpg"> = <img src="images/JessicaFaceRectangleMasked.jpg">
                **OR**
        <img src="images/jpShrunk.png"> + <img src="images/CircleMaskForJessicaFace"> <img src="images/JessicaFaceCircleMasked">