Code implementing Deep-learning using OpenCV as taught in Deep-learning for computer vision with Python(Starter Bundle).

Common layer pattern
	INPUT(Square dimensional) => [[CONV => RELU]*N => POOL?]*M => [FC => RELU]*K => FC


Variations in experiments
1. Pooling: Use overlapping pooling for large dimensional inputs. Try non-overlapping pooling. Also, try to use convolutions with a large stride instead of a pooling layer.
2. Activation functions: Try ReLU first. Compare using Leaky ReLU and ELU to see if you get better results
3. Batch normalization: use Batch normalization after the activation layer. Test difference if used before activation. Do not apply before the softmax classifier
4. Dropout: Dropout with p=0.5 in between an FC that outputs softmax probabilities. Use dropout p = [0.10, 0.25] after downsampling(pooling or convolution with stride>1)

Common heuristics(CNN).
1. Input image should be square(platform can use linear optimization libraries to train faster).
2. Input layer dimensions should be divisible by  mulitple times after the first conv is done. This allows POOL to be done more efficiently
3. Only use large filter size at the beginning of networks with a very large dimensionality. Filters of 3*3 or 5*5 are preferrable later in the network.
4. Applying zero padding when stacking COnv layers often improves accurracy.
5. Pooling layers with 3* 3 kernels are only used rarely and at the beginning of a network. it is more common to use 2*2 kernels with a stride of 2.
6. Batch normalization increases the training time but it stabilizes training and makes it easier to tune other hyperparameters.
7. You need to explicitly train you network to be rotation and scale invariant. A CNN is naturally translational(location withing image) invariant