# FCRN：Fully Convolutional Residual Network for Depth Estimation
A Pytorch implementation of Laina, Iro, et al. "Deeper depth prediction with fully convolutional residual networks." 3D Vision (3DV), 2016 Fourth International Conference on. IEEE, 2016.

The original implementation is in TensorFlow(https://github.com/iro-cp/FCRN-DepthPrediction).

To running, please do the following steps:

 1. Download the NYU Depth Dataset V2 Labelled Dataset : http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat.
	
2. Download the pre-trained TensorFlow weights as a .npy file for a part of the model from Laina et al. from http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy.
	
3. Put the above two files in the same directory as the code.
			
4. Run train.py to train, run test.py to evaluate the result.

Some of the results(rgb image, ground truth, predicted depth):
				
![image](https://github.com/XPFly1989/FCRN/blob/master/result_521.png)
![image](https://github.com/XPFly1989/FCRN/blob/master/result_599.png)
