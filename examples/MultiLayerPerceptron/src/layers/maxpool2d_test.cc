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
const std::string filename = test_data_folder + "maxpool_VALID.pb";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"Max Pool"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer maxpool_layer;
    // Read data
    std::fstream input_stream(filename, std::ios::in | std::ios::binary);
    if (!maxpool_layer.ParseFromIstream(&input_stream)) {
        std::cerr << "Failed to parse protobuf." << std::endl;
        return -1;
    }

    bool is_pass=true;

    int batch=maxpool_layer.input().shape(0);
    int input_h=maxpool_layer.input().shape(1);
    int input_w=maxpool_layer.input().shape(2);
    int channels=maxpool_layer.input().shape(3);
    int input_size=maxpool_layer.input().data_size();
    int in_shape[4]={batch, input_h, input_w, channels};
    int shape[4]={batch, input_h, input_w, channels};
    int input[BUFFER_SIZE];
    for(int i=0; i<input_size; ++i){
        input[i] = maxpool_layer.input().data(i);
    }
    
    int kernel_h=maxpool_layer.kernel().shape(1);
    int kernel_w=maxpool_layer.kernel().shape(2);
    int k_shape[4]={1, kernel_h, kernel_w, 1};
    int stride=maxpool_layer.kernel().strides(1);
    
    int output_h=maxpool_layer.output().shape(1);
    int output_w=maxpool_layer.output().shape(2);
    int output_c=maxpool_layer.output().shape(3);
    int output_size=maxpool_layer.output().data_size();
    int gold_output[BUFFER_SIZE];
    for(int i=0; i<output_size; ++i){
        gold_output[i] = maxpool_layer.output().data(i);
    }
    
    int output_shape[4];
    maxpool2d_shape(output_shape, in_shape, k_shape, stride);
    // forward pass
    int output[BUFFER_SIZE];
    int grad_map[BUFFER_SIZE]; // contains information for backward pass
    maxpool2d<int>(output, output_shape, grad_map, input, in_shape, k_shape, stride);
    
    // check shape
    for(int i=0; i<4; ++i) {
        if(output_shape[i] != maxpool_layer.output().shape(i)) {
            throw std::logic_error("Max_Pooling output shape is wrong!");
        }
    }
    compare_buffers<int>(is_pass, output, gold_output, output_size, "Maxpool forward pass");

    // backprop
    int gold_grad[BUFFER_SIZE];
    for(int i=0; i<maxpool_layer.grad_backprop().data_size(); ++i){
        gold_grad[i] = maxpool_layer.grad_backprop().data(i);
    }

    maxpool2d_backprop<int>(output, in_shape, grad_map, gold_output, output_shape);
    compare_buffers<int>(is_pass, output, gold_grad, input_size, "Maxpool backprop");

    //print_tensor(output, in_shape);
    //print_tensor(gold_grad, in_shape);

    print_pass(is_pass);
    
    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
