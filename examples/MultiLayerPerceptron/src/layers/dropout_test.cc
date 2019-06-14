#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <math.h>
#include <exception>
#include <eigen3/Eigen/Dense>
#include "Layer.pb.h"
#include "layers.h"
#include "tensor.h"
#include "utils/error_checking.h"
#include <limits>
#include "test_utils.h"

static const int BUFFER_SIZE=100000;

const std::string test_data_folder="test_data/";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::cout<<"Dropout"<<std::endl;
    std::cout<<"############################"<<std::endl;
    bool is_pass = true;

    std::srand(42);

    float keep_prob=0.5;
    const float factor=10;

    int batch=8;
    int height=10;
    int width=10;
    int channels=8;
    int shape[4]={batch, height, width, channels};
    int size = batch * height * width * channels;

    float input[BUFFER_SIZE];
    int map[BUFFER_SIZE];
    float output[BUFFER_SIZE];

    for(int i=0; i<size; ++i){
        float random = (float)std::rand()/(float)RAND_MAX;
        input[i]=(random-0.5)*2*factor;
    }

    dropout(output, map, input, shape, keep_prob);
    int zeros=0;
    for(int i=0; i<size; ++i){
        if(output[i]==0) ++zeros;
    }
    if(zeros>size*keep_prob*1.25 || zeros<size*keep_prob*0.75) {
        std::cout<<"Dropout forward pass probably wrong."<<std::endl;
        is_pass=false;
    }
    else {
        std::cout<<"Dropout forward pass successful."<<std::endl;
    }

    dropout_backprop(output, map, input, shape, keep_prob);
    int errors=0;
    for(int i=0; i<size; ++i){
        if( map[i]==0 && output[i]!=0) ++errors;
    }
    if(errors>0) {
        std::cout<<"Dropout backprop failed with "<<errors<<" errors."<<std::endl;
        is_pass=false;
    }
    else {
        std::cout<<"Dropout backprop successful."<<std::endl;
    }

    //print_tensor(output, shape);
    //print_tensor(map, shape);
    //print_tensor(input, shape);

    print_pass(is_pass);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
