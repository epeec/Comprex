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
const std::string filename = test_data_folder + "softmax.pb";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"Softmax"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer softmax_layer;
    // Read data
    std::fstream input_stream(filename, std::ios::in | std::ios::binary);
    if (!softmax_layer.ParseFromIstream(&input_stream)) {
        std::cerr << "Failed to parse protobuf." << std::endl;
        return -1;
    }

    bool is_pass=true;

    Tensor<float> input(softmax_layer.input().data(), softmax_layer.input().shape());
    Tensor<float> gold_output(softmax_layer.output().data(), softmax_layer.output().shape());
    Tensor<float> gold_grad(softmax_layer.grad_backprop().data(), softmax_layer.grad_backprop().shape());

    Tensor<float> output(softmax_layer.input().shape());

    softmax<float>(output.data(), input.data(), input.shape());
    compare_buffers<float>(is_pass, output.data() , gold_output.data(), output.get_size(), "Softmax forward pass");

    softmax_backprop<float>(output.data(), gold_output.data(), gold_output.data(), gold_output.shape());
    compare_buffers<float>(is_pass, output.data(), gold_grad.data(), gold_grad.get_size(), "Softmax backprop");

    //output.print();
    //gold_grad.print();

    print_pass(is_pass);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
