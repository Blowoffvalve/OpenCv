#Code implementing Deep-learning using OpenCV

##Common layer pattern
	INPUT(Square dimensional) => [[CONV => RELU]*N => POOL?]*M => [FC => RELU]*K => FC


##Variations in experiments
1. Pooling: Use overlapping pooling for large dimensional inputs. Try non-overlapping pooling. Also, try to use convolutions with a large stride instead of a pooling layer. Some networks do not use pooling at all(dimensionality reduction only via Conv2D with large strides). Explore using this kind of architecture.
2. Activation functions: Try ReLU first. Compare using Leaky ReLU and ELU to see if you get better results
3. Batch normalization: use Batch normalization after the activation layer. Test difference if used before activation. Do not apply before the softmax classifier
4. Dropout: Dropout with p=0.5 in between an FC that outputs softmax probabilities. Use dropout p = [0.10, 0.25] after downsampling(pooling or convolution with stride>1)
5. Learning Rate Scheduling: Try to not use a LR to observe what your baseline accuracy is. Then use various learning rates scheduling with various decay factors to observe your validation accuracy.


##Common heuristics.
1. Input image should be square(platform can use linear optimization libraries to train faster).
2. Input layer dimensions should be divisible by  mulitple times after the first conv is done. This allows POOL to be done more efficiently
3. Only use large filter size at the beginning of networks with a very large dimensionality. Filters of 3*3 or 5*5 are preferrable later in the network.
4. Applying zero padding when stacking COnv layers often improves accurracy.
5. Pooling layers with 3* 3 kernels are only used rarely and at the beginning of a network. it is more common to use 2*2 kernels with a stride of 2.
6. Batch normalization increases the training time but it stabilizes training and makes it easier to tune other hyperparameters. Very highly recommended.
7. You need to explicitly train you network to be rotation and scale invariant(use random rotation and scaling during training). A CNN is naturally translational(location withing image) invariant.
8. Scheduling the learning rate to reduce periodically helps reduce overfitting. You can either use a learning rate scheduler (with the decay set to lr/#epochs) or use a stepbased scheduler callback object(see appendix 1 below)
9. Watch for underfitting(check for training loss not going low) and overfitting( by checking that the training and test losses are wildly divergent). If validation loss is increasing, you are clearly overfitting. 
10.Use regularization(weight decay, dropout, data augumentation, early stopping...) to control overfitting
11.Monitor the training and validation loss using a training monitor callback(See appendix 2)
12.Saving the weights as validation loss improves(using keras.callbacks.ModelCheckpoint) allows you to persist high-performing networks to disk. The devil is a bastard.
13.Use keras.plot_model() to visualize models you've created. It shows dimensions and their order.
14.When working with image data and you intend to do manual feature extraction(For example to pickup numbers), the sequence of extract regions of interest is in reference Appendix 3.


##Appendix
1. Callback sample for configurable learning rate decay

	def step_decay(epoch):
		#initialize the base learning rate, drop factor and the number of epochs after which the epoch should be dropped
		initAlpha = 0.01
		factor = 0.25
		dropEvery = 5
		
		#Calculate learning rate for the current epoch
		alpha = initAlpha * (factor ** np.floor((1+epoch)/dropEvery))
		
		#return the learning rate.
		return float(alpha)
	
	callbacks = [LearningRateScheduler(step_decay)]
	
	model.fit(trainX, trainY, validation_data = (testX, testY), batch_size=64, epochs=40, callbacks = callbacks)

2. callback sample for monitoring training using the process ID as the identifier
	import os
	from trainingmonitor import TrainingMonitor
	#download this code snippet for the training monitor class  https://github.com/Blowoffvalve/OpenCv/blob/master/DL4CV/utilities/callbacks/trainingmonitor.py
	
	#define the output directory
	outputdir = "/output"
	figPath = os.path.sep.join([outputdir, "{}.jpg".format(os.getpid())])
	jsonPath = os.path.sep.join([outputdir, "{}.json".format(os.getpid())])
	callbacks = [TrainingMonitor(figPath,jsonPath=jsonPath)]
	model.fit(trainX, trainY, validation_data = (testX, testY), batch_size=64, epochs=100, callbacks = callbacks)
	
3. 	Manual feature extraction process
		A. Convert to grayscale(1 channel)
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		B. Add a border padding so the image isn't at the edge.
				gray = cv2.copyMakeBorder(gray, 20,20,20,20,cv2.BORDER_REPLICATE)
		C. Use a thresholding function to scale the image from 0 to 1. Pytorch requires that the images be in the range -1 to 1 for better convergence.
				cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]
		D. Extract the contours and select a fixed number of them(i select 4 here)
				cnts = cv2.findContours(thresh, mode = cv2.RETR_EXTERNAL method=cv2.CHAIN_APPROX_SIMPLE)
				cnts = cnts[0] if imutils.is_cv2() else cnts[1]
				cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
				cnts = contours.sort_contours(cnts)[0]
		E. Loop over the contours and compute the regions of interests(roi)
				for c in cnts:
					#compute the bounding box for the contour then extract the digit
					(x, y, w, h) = cv2.boundingRect(c)
					roi = gray[y-5:y+h+5, x-5:x+w+5]
					roi = preprocess(roi, 28, 28)
					roi = np.expand_dims(img_to_array(roi), axis=0)/255.0
		F. Utilize the rois for whatever you want e.g. to predict the most likely probability
					pred = model.predict(roi).argmax(axis=1)