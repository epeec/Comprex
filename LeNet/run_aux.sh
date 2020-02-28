#!/bin/bash -i

cd /home/loroch/PROJECTS/tensorflow-gpi/LeNet
#source /home/loroch/.bashrc

conda activate tensorflow-1.12
#export PATH=${PATH}:/opt/cuda/cuda_9/bin:opt/cuda/include

#echo `hostname` $LD_LIBRARY_PATH
kernprof -v -l -o `hostname`.prof main.py
#python main.py
