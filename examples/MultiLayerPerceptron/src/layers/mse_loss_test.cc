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
const std::string filename = test_data_folder + "mse.pb";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"MSE"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer mse_layer;
    // Read data
    std::fstream input_stream(filename, std::ios::in | std::ios::binary);
    if (!mse_layer.ParseFromIstream(&input_stream)) {
        std::cerr << "Failed to parse protobuf." << std::endl;
        return -1;
    }

    bool is_pass=true;

    const float EPS = 1e-4;//std::numeric_limits<float>::min()*16;
    int batch = mse_layer.input().shape(0);
    int width = mse_layer.input().shape(1);
    int shape[4] = {batch, width, 1, 1};
    int size = batch*width;

    float gold_output = mse_layer.output().data(0);

    float input[BUFFER_SIZE];
    float labels[BUFFER_SIZE];
    float gold_grad[BUFFER_SIZE];
    for(int i=0; i<size; ++i){
        input[i]=mse_layer.input().data(i);
        labels[i]=mse_layer.kernel().data(i);
        gold_grad[i]=mse_layer.grad_backprop().data(i);
    }

    float output[BUFFER_SIZE];
    mse_loss(output, input, labels, shape);
    compare_buffers<float>(is_pass, output, &gold_output, 1, "MSE loss");

    mse_loss_backprop<float>(output, input, labels, shape);
    compare_buffers<float>(is_pass, output, gold_grad, size, "MSE backprop");

    //print_vector<float>(output, shape);
    //print_vector<float>(gold_grad, shape);

    print_pass(is_pass);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
