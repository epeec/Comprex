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
const std::string filename = test_data_folder + "add_bias.pb";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"Add Bias"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer bias_layer;
    // Read data
    std::fstream input_stream(filename, std::ios::in | std::ios::binary);
    if (!bias_layer.ParseFromIstream(&input_stream)) {
        std::cerr << "Failed to parse protobuf." << std::endl;
        return -1;
    }

    bool is_pass=true;
    bool is_flat=false;

    Tensor<int> input(bias_layer.input().data(), bias_layer.input().shape());
    Tensor<int> kernel(bias_layer.kernel().data(), bias_layer.kernel().shape());
    Tensor<int> gold_output(bias_layer.output().data(), bias_layer.output().shape());
    Tensor<int> gold_filter_grad(bias_layer.grad_filter().data(), bias_layer.grad_filter().shape());
    Tensor<int> gold_backprop_grad(bias_layer.grad_backprop().data(), bias_layer.grad_backprop().shape());

    Tensor<int> output(bias_layer.input().shape());

    add_bias<int>(output.data(), input.data(), kernel.data(), input.shape(), is_flat);
    compare_buffers<int>(is_pass, output.data() , gold_output.data(), output.get_size(), "Add Bias forward pass");

    add_bias_backprop_grad<int>(output.data(), gold_output.data(), output.shape(), is_flat);
    compare_buffers<int>(is_pass, output.data(), gold_backprop_grad.data(), gold_backprop_grad.get_size(), "Add Bias backprop gradient");

    add_bias_filter_grad<int>(output.data(), gold_output.data(), output.shape(), is_flat);
    compare_buffers<int>(is_pass, output.data(), gold_filter_grad.data(), gold_filter_grad.get_size(), "Add Bias filter gradient");

    //output.print();
    //gold_filter_grad.print();

    print_pass(is_pass);

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
