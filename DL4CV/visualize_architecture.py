#import the necessary packages
from utilities.nn.conv.lenet import LeNet
from keras.utils import plot_model

#initialize and write the architechture visualization graph to disk(Note that you need to have installed graphviz(both the pip package and OS application) and pydot-ng)
model = LeNet.build(28,28,1,10)
plot_model(model, to_file = "lenet.png", show_shapes=True)