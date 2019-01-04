from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    """
    The dataFormat can either be 'channels_first' i.e. d h * w or 'channels_last' h * w * d. if set to None, it uses the keras default dataFormat specified in ~/.keras/keras.json.
    """
    def __init__(self, dataFormat = None):
        #store the image data format.
        self.dataFormat = dataFormat
        
    def preprocess(self, image):
        #apply the kerast img_to_array function that rearranges the dimensions of the image.
        return img_to_array(image, data_format=self.dataFormat)