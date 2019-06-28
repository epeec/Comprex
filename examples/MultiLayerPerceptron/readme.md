# Multilayer Perceptron Example

This example trains a three layer multilayer perceptron (input-output: 784-128-64-10) on the MNIST dataset. It uses synchronous, distributed training, based on the Comprex communication library. The model size is kept small and the number of training epochs is low, so the entire training should only take a few minutes. The final accuracy is computed on a seperate test dataset.
The hyperparameters of this example were not tuned for the best results. Also, other hyperparameters change with the number of training ranks. 


## Requirements
This example needs Comprex to be installed. Refer to the readme of the parent directory to see installation guidelines.

The MNIST dataset is needed. Download and unzip MNIST from
 
http://yann.lecun.com/exdb/mnist/

## Setup
Open the file "CMakeLists.txt" and make sure the directories
```
set (MNIST_PATH "xxx/DATASETS/MNIST")
set (GTEST_ROOT "xxx/googletest")
set (GASPI_CXX_ROOT "xxx/GaspiCxx")
set (COMPREX_ROOT "xxx/comprex")
```
are set to the MNIST dataset, googletest, GaspiCxx and Comprex directories, respectively.

It is recommended to create a "build" directory in this directory and to run 
```
cmake ..
```
from there. This will set up the project makefiles.

In order to make the program run, you will require a nodelist inside the build directory. It is sufficient to add a file named 'nodelist' inside the build directory with the content
```
localhost
localhost
```
where the number of 'localhost' entries corresponds to computing ranks and may be customized.

## Compiling and Running
Compile the project with 
```
make main
```
from the build directory.

Run the project with
```
make gaspi_run
```

## Changing Training Hyperparameters
Changing the training hyperparameters requires to change the source code in "src/main.cc". The main hyperparameters are defined in the very beginning of the file as static variables. 

The threshold for the comprex optimizer is set in the file "src/optimizer.h", in the constructor of "Comprex_Optimizer"
```
this->comprex_threshold = 0.1;
```
