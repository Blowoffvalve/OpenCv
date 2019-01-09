Code implementing Deep-learning using OpenCV as taught in Deep-learning for computer vision with Python(Starter Bundle).

Common layer pattern
	INPUT(Square dimensional) => [[CONV => RELU]*N => POOL?]*M => [FC => RELU]*K => FC


Variations in experiments
1. Pooling: Use overlapping pooling for large dimensional inputs. Try non-overlapping pooling. Also, try to use convolutions with a large stride instead of a pooling layer. Some networks do not use pooling at all(dimensionality reduction only via Conv2D with large strides). Explore using this kind of architecture.
2. Activation functions: Try ReLU first. Compare using Leaky ReLU and ELU to see if you get better results
3. Batch normalization: use Batch normalization after the activation layer. Test difference if used before activation. Do not apply before the softmax classifier
4. Dropout: Dropout with p=0.5 in between an FC that outputs softmax probabilities. Use dropout p = [0.10, 0.25] after downsampling(pooling or convolution with stride>1)
5. Learning Rate Scheduling: Try to not use a LR to observe what your baseline accuracy is. Then use various learning rates scheduling with various decay factors to observe your validation accuracy.


Common heuristics.
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


Appendix
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