# Classification-of-benign-and-malignant-lesions
Code for paper "Diagnostic efficiency of multi-modal MRI based deep learning with Sobel operator in differentiating benign and malignant breast mass lesions"

# requirements
The main packages required are Pytorch,SimpleITK,opencv-python, scikit-learn and so on.

# How to use
Considering that the network involves single-modal and multi-modal recognition, different dataset classes, network classes and train classes are written. *1m* in the suffix indicates the single-modal version, *4m*, *8m* and *mm* indicate the multi-modal version. The **utils** folder contains the datasets class of the model and some metrics functions and data preprocessing code. The **net** folder contains networks class of the model. The **data** folder contains dataset of MRI of BC. In train class, you should provide adequate parameters like root, path of train file , path of model save and so on. 
