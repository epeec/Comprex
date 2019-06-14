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
const std::string filename = test_data_folder + "smce.pb";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"Softmax-Cross-Entropy Loss"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer smce_layer;
    // Read data
    std::fstream input_stream(filename, std::ios::in | std::ios::binary);
    if (!smce_layer.ParseFromIstream(&input_stream)) {
        std::cerr << "Failed to parse protobuf." << std::endl;
        return -1;
    }

    bool is_pass=true;

    int batch = smce_layer.input().shape(0);
    int width = smce_layer.input().shape(1);
    int shape[4] = {batch, width, 1, 1};
    int res_shape[4] = {batch, 1, 1, 1};
    int size = batch*width;

    float gold_loss=0;
    float gold_output[BUFFER_SIZE]; 
    for(int i=0; i<smce_layer.output().data_size(); ++i) {
        gold_output[i] = smce_layer.output().data(i);
        gold_loss += gold_output[i];
    }
    gold_loss /= batch;


    float input[BUFFER_SIZE];
    float labels[BUFFER_SIZE];
    float gold_grad[BUFFER_SIZE];
    for(int i=0; i<size; ++i){
        input[i]=smce_layer.input().data(i);
        labels[i]=smce_layer.kernel().data(i);
        gold_grad[i]=smce_layer.grad_backprop().data(i);
    }

    float output[BUFFER_SIZE];
    float grad[BUFFER_SIZE];
    smce_loss(output, grad, input, labels, shape);
    compare_buffers<float>(is_pass, output, &gold_loss, 1, "SMCE loss");
    //std::cout<<output[0]<<" "<<gold_loss<<std::endl;
    compare_buffers<float>(is_pass, grad, gold_grad, size, "SMCE backprop1");

    smce_loss_backprop<float>(grad, input, labels, shape);
    compare_buffers<float>(is_pass, grad, gold_grad, size, "SMCE backprop2");

    //print_vector<float>(output, shape);
    //print_vector<float>(gold_grad, shape);

    print_pass(is_pass);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
