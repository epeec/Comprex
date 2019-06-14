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
const std::string filename = test_data_folder + "fc.pb";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"Fully Connected"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer fc;
    // Read data
    std::fstream input(filename, std::ios::in | std::ios::binary);
    if (!fc.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse protobuf." << std::endl;
        return -1;
    }
    bool is_pass=true;
    int error_count=0;
    const float EPS = 1e-3;
    
    int k_height=fc.kernel().shape(0);
    int k_width=fc.kernel().shape(1);
    int k_size = k_height * k_width;
    
    int i_width = k_height;
    int batch=fc.input().shape(0);
    int i_size = i_width*batch;
    
    int result_size = fc.output().data_size();
    
    int weights[BUFFER_SIZE];
    for(int i=0; i<k_size; ++i)
        weights[i]=fc.kernel().data(i);
    int inputs[BUFFER_SIZE];
    for(int i=0; i<i_size; ++i)
        inputs[i]=fc.input().data(i);
    int gold_result[BUFFER_SIZE];
    for(int i=0; i<result_size; ++i)
        gold_result[i]=fc.output().data(i);
    int gold_update[BUFFER_SIZE];
    for(int i=0; i<k_size; ++i)
        gold_update[i]=fc.grad_filter().data(i);
    int gold_gradient[BUFFER_SIZE];
    for(int i=0; i<i_size; ++i)
        gold_gradient[i]=fc.grad_backprop().data(i);

    int fc_result[BUFFER_SIZE];
    int weight_shape[2]={k_height,k_width};
    int input_shape[4]={batch,i_width,1,1};
    int output_shape[4]={batch, k_width,1,1};

    // Forward pass
    fully_connected<int>(fc_result, inputs, input_shape, weights, weight_shape);
    error_count=0;
    for(int i=0; i<result_size;++i) {
        if(fc_result[i] != gold_result[i])
            ++error_count;
    }
    if(error_count==0) {
        std::cout<<"Forward pass successful."<<std::endl;
    }
    else {
        std::cout<<"Forward pass failed with "<<error_count<<" errors."<<std::endl;
        is_pass=false;
    }

    // Backprop filter update
    fc_filter_gradient<int>(fc_result, gold_result, output_shape, inputs, input_shape);
    error_count=0;
    for(int i=0; i<k_size;++i) {
        if(fc_result[i] != gold_update[i]){
            ++error_count;
        }
    }
    if(error_count==0) {
        std::cout<<"Filter update successful."<<std::endl;
    }
    else {
        std::cout<<"Filter update failed with "<<error_count<<" errors."<<std::endl;
        is_pass=false;
    }

    // Backprop gradient update
    fc_backprop_gradient<int>(fc_result, gold_result, output_shape, weights, weight_shape);
    error_count=0;
    for(int i=0; i<i_size;++i) {
        if(fc_result[i] != gold_gradient[i]){
            ++error_count;
        }
    }
    if(error_count==0) {
        std::cout<<"Gradient backprop successful."<<std::endl;
    }
    else {
        std::cout<<"Gradient backprop failed with "<<error_count<<" errors."<<std::endl;
        is_pass=false;
    }
    
    print_pass(is_pass);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
