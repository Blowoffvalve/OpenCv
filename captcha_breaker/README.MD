##Code to create a download images that contain only numbers, label them and train a model using that data

###download_images.py retrieves the images
###annotate.py gamifys the laborious process of creating the dataset from the images. It splits each image into it's constituent numbers, and shows you each number. entering the number moves that images split into the /dataset/{number} directory.
###train.py trains the LeNet network with this data and save the network to a specified directory
###test.py lets you specify an image and a network to get the network's prediction of what the image contains
