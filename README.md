# ComprEx

## Introduction
ComprEx is a GaspiCxx based application library, which exchanges large 
vectors with lossy compression and local error accumulation.

## Installation

### Requirements:
- `cmake` version > 3.6
- `c++ 14` (presently with `gcc-5.2.0`)
- `GPI` as a prerequisite for GaspiCxx, available [here](https://github.com/cc-hpc-itwm/GPI-2)
- `GaspiCxx` available [here](https://github.com/cc-hpc-itwm/GaspiCxx)
- (optional) [Tensorflow](https://www.tensorflow.org/) version 2.3, it is strongly recommended to use a [conda](https://docs.conda.io/en/latest/) environment

### Building Prerequisites

- Install `GPI` before attempting to install `GaspiCxx`.
- Make sure `GaspiCxx` is build with the `BUILD_SHARED_LIBS` option `on`. Please modify the entry in the main `CMakeLists.txt` of `GaspiCxx`! Comprex will search for the `GaspiCxx` library in the `build/src/` directory of `GaspiCxx`.
- If `Tensorflow` should be used, it is recommended to use a `conda` environment.

### Building ComprEx

1. Clone the git repository.
    ``` 
    git clone https://github.com/epeec/Comprex.git
    ```

2. The main `CmakeLists.txt` needs to be updated with the installation path of `GaspiCxx`.
    ```
    set (GASPI_CXX_ROOT "<GaspiCxx directory>")
    ```
    GPI will be detected automatically.
    If Tensorflow operations should be installed as well, then
    set the option for `BUILD_TFOPS` to `ON`.
    If Infiniband is used, switch the `LINK_IB` to `ON`.

3. Build and install the Comprex library plus examples.
    Make sure the correct conda environment is activated, if Tensorflow operations should be installed.
    In the main directory of Comprex execute
    ```
    mkdir build
    cd build
    cmake ..
    make install
    ```
    If successful,
    - the library `libPyGPI.so` is installed in `lib/`
    - optionally, the library `libtfGPIOps.so` is installed in `lib/`, if `BUILD_TFOPS` was `ON`.
    
    Additionally, a dummy `nodefile` is created in the `build` directory for the examples and tests. Also, a series of shell files for various examples and tests are created in `build`.

    The include files can be found in the Comprex main directory `include/` folder.

## Running Examples
    
The Examples are set up to run without further configuration after the installation. You can run the following commands from the `build` dirtectory to try out the examples:
```
make run_example_gaspiEx_c
make run_example_gaspiEx_py
```
Should finish with a `+++ PASSED +++` message.
```
make run_example_comprEx_c
make run_example_comprEx_py
```
Should finish with a `+++ PASSED +++` message.
```
make run_example_allreduce_speed_c
```
Should perform a few timing measurements and print the results on screen.
```
make run_example_comprex_speed_c
```
Should perform a few timing measurements and print the results on screen.

## Running distributed deep learning training with Tensorflow

In the `LeNet` directory, there is a script prepared to run the Training of the Lenet model on the MNIST dataset. The dataset will be acquired automatically. The `nodelist` is configured to run two ranks on the local machine.
Make sure the correct conda environment is activated and run
```
source run.sh
```
Lenet will be trained with several setups. The results will be written in `log/`.
The results can be inspected by going into the `log/` directory and starting Tensorboard with
```
tensorboard --logdir=.
```
In the Chrome webbrowser, visit `http://localhost:6006/` to see the training results.