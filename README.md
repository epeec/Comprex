# Comprex

This is the file repository for the Comprex compressed communication library.

There are two files in this folder:
* LeNet - contains a Tensorflow Keras example of distributed DNN training with Comprex
* pyGPI - contains the Comprex communication library and other necessary GPI code

## Installation

This project has the following dependencies (higher versions might work as well):

* Python 3.6.7
* gcc 8.3.0
* cmake 3.13.4
* [GPI-2](https://github.com/cc-hpc-itwm/GPI-2) 1.4.0 (compile with -fpic!)
* [GaspiCxx](https://github.com/cc-hpc-itwm/GaspiCxx) (compile with -fpic!)

For the LeNet example:

* Tensorflow 1.12 ( tested on GPU version )

IMPORTANT: It is necessary that GPI-2 and GaspiCxx are compiled with the -fpic flag!

Install the required software according to the respective documentations.

1) Make sure the dependencies above are all set up correctly. GPI-2 and GaspiCxx need to be compiled with the -fpic compiler flag.

2) In the pyGPI directory, modify the content of the 'CMakeLists.txt' in the 'USER SECTION'. Specifically, set the 'GTEST_ROOT' and 'GASPI_CXX_ROOT' with the path to your googletest and your GaspiCxx installation directory, respectively.

3) Go into the pyGPI folder. Set up the build environment with:
```bash
mkdir build
cd build
cmake ..
```
Notice that some shellscripts 'gaspi_*.sh' are created in the 'build' directory as well. They are used to start the python examples with 'gaspi_run'.

4) Compile the source files. In the 'build' directory, execute
```bash
make
```

The library file 'libPyGPI.so' is generated in the 'build/src' directory, which is used by Python. The installation is now complete.

## Running the pyGPI examples

In this section, it is assumed that the user is in the 'pyGPI/build' directory.

When you have a multi-node session, you need to create a machinefile (aka hostfile). Refer to the GPI documentation on how to create a machinefile for the 'gaspi_run' command. The machinefile must be located in the 'build' directory and it must be named 'nodelist'.

In the provided example, rank 0 sends a vector of integer data multiple times to rank 1. Rank 1 accumulates all the received values. Finally, rank 0 flushes out its residual vector and rank 1 accumulates that as well. The resulting vector on rank 1 should be the original vector times the number of communication rounds. The results are checked automatically and a '+++PASSED+++' message is printed in case of success.

In order to run the example, type
```bash
make gaspi_example1_cpp
```
for the C++ based example, or
```bash
make gaspi_example1_py
```
for the Python based example. In the Python example, you can see additional output by typing
```bash
gaspi_logger&
```
before running the Python example.

Both examples should finish with a '+++PASSED+++' message.

## Running the LeNet example

This example runs on Tensorflow 1.12, however Comprex itself is not bound to any Tensorflow version.

It is assumed that the user is in the 'LeNet' directory and that Comprex has been installed successfully.

The user needs to provide a hostfile with the names of the nodes. The 'run_aux.sh' file needs to be modified, such that the remote host changes to the correct directory and the right environment is set up for Tensorflow (e.g. a 'conda activate <my_environment>').

The training is started with 
```bash
source run.sh
```
The MNIST dataset should be downloaded and prepared automatically when the script is run for the first time. The model trains on MNIST for a single epoch. At the end of the training, the model should be evaluated and the test accuracy is printed for each rank. The test accuracy should be above 90% for each rank.