# Image_recognition
Image classification of plants

Python libraries to be installed://
sudo pip3 install matplotlib
sudo pip3 install keras
sudo pip3 install tensorflow

Link to run tensorflow with GPU
https://www.tensorflow.org/install/gpu

-You have to install compatible controller versions of nvidia and cuda.


The images dataset are divided into different folders in the Documents folder.

Image dataset of kaggle/plantcv/Imageclef images
  -Training data of plants 102k : training_dataset_y+
  -Training data of non-plants 101.112k : training_dataset_y-
  -Validation data of plants 20.4k : validation_dataset_y+
  -Validation data of non-plants 20.4k : validation_dataset_y-
  
Image dataset of the fablab
  -Training data of plants 157 : training_my_dataset_y+
  -Training data of non-plants 173 : training_my_dataset_y-
  
Python files in folder Documents/Classification/TORUN

classifier2.py --> Configuration only with internet datasets
Layers parameters: 2 Convolution layers depth of the output feautured map 16, size of the patches extracted of the inputs 3x3 and 2 pooling layers
Hyperparameters: batch size = 120 for training and 19 for validation & epoch = 15
                  


