#import packages

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    """
    INPUT => CONV(5*5, 20) => RELU => POOL => CONV(5*5, 50) => RELU => POOL(2*2) => FC => RELU => FC 
    """
    @staticmethod
    def build(width, height, depth, classes):
        #initialize the model
        model = Sequential()
        inputShape=(height, width, depth)
        
        #If we are using 'channels first' put channels first
        if K.image_data_format()== "channels_first":
            inputShape = (depth, height, width)
        
        #first set of Conv=> RELU => Pool layers
        model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #Second set of layers CONV(5*5, 50) => RELU => POOL(2*2)
        model.add(Conv2D(50, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
        
        #SIngle Fully connected layer
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
        