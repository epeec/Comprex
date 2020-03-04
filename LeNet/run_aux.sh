#!/bin/bash -i

# change current directory to 'LeNet' 
cd /home/loroch/PROJECTS/tensorflow-gpi/LeNet

# If required, set up the environment
conda activate tensorflow-1.12

# start the DNN training and evaluation
python main.py