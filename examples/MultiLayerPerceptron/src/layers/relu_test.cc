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
const std::string filename = test_data_folder + "relu.pb";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"ReLU"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer relu_layer;
    // Read data
    std::fstream input_stream(filename, std::ios::in | std::ios::binary);
    if (!relu_layer.ParseFromIstream(&input_stream)) {
        std::cerr << "Failed to parse protobuf." << std::endl;
        return -1;
    }

    bool is_pass=true;
    int error_count;

    int batch=relu_layer.input().shape(0);
    int width=relu_layer.input().shape(1);
    int shape[4]={batch,width,1,1};
    int size=relu_layer.input().data_size();

    int input[BUFFER_SIZE];
    for(int i=0; i<relu_layer.input().data_size(); ++i)
        input[i]=relu_layer.input().data(i);
    
    int gold_output[BUFFER_SIZE];
    for(int i=0; i<relu_layer.output().data_size(); ++i)
        gold_output[i]=relu_layer.output().data(i);

    int gold_gradient[BUFFER_SIZE];
    for(int i=0; i<relu_layer.grad_backprop().data_size(); ++i)
        gold_gradient[i]=relu_layer.grad_backprop().data(i);

    int output[BUFFER_SIZE];
    relu<int>(output, input, shape);
    error_count=0;
    compare_buffers<int>(is_pass, output, gold_output, size, "ReLU forward pass");

    relu_backprop<int>(output, gold_output, input, shape);
    error_count=0;
    compare_buffers<int>(is_pass, output, gold_gradient, size, "ReLU backprop");
    
    print_pass(is_pass);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
