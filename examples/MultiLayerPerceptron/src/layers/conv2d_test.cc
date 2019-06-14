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
//const std::string filename = test_data_folder + "conv_valid_ns.pb";
const std::string filename = test_data_folder + "conv_tiny_same_s.pb";
//const std::string filename = test_data_folder + "conv_small3.pb"; 
//const std::string filename = test_data_folder + "conv_same.pb"; 

int main() {
    
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"Convolution"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer conv; 
    // Read data
    std::fstream input_stream(filename, std::ios::in | std::ios::binary);
    if (!conv.ParseFromIstream(&input_stream)) {
        std::cerr << "Failed to parse protobuf." << std::endl; 
        return -1;
    }
    std::cout<<"Convolution test-file:"<<std::endl;
    //Kernel
    std::cout<<"Kernel: ";
    std::cout<<"[";
    for(int dim=0; dim < conv.kernel().shape_size(); ++dim){
        std::cout << conv.kernel().shape(dim) << ", ";
    } std::cout<<"\b\b]"<<std::endl;
    std::cout<<std::endl;
    //Input
    std::cout<<"Input: ";
    std::cout<<"[";
    for(int dim=0; dim<conv.input().shape_size(); ++dim){
        std::cout<<conv.input().shape(dim)<<", ";
    } std::cout<<"\b\b]"<<std::endl;
    std::cout<<std::endl;
    //Output
    std::cout<<"Output: ";
    std::cout<<"[";
    for(int dim=0; dim<conv.output().shape_size(); ++dim){
        std::cout<<conv.output().shape(dim)<<", ";
    } std::cout<<"\b\b]"<<std::endl;
    std::cout<<std::endl;
    //Gradient of filters
    std::cout<<"Filter gradient: ";
    std::cout<<"[";
    for(int dim=0; dim<conv.grad_filter().shape_size(); ++dim){
        std::cout<<conv.grad_filter().shape(dim)<<", ";
    } std::cout<<"\b\b]"<<std::endl;
    std::cout<<std::endl;
    //Gradient of backprop
    std::cout<<"Backprop gradient: ";
    std::cout<<"[";
    for(int dim=0; dim<conv.grad_backprop().shape_size(); ++dim){
        std::cout<<conv.grad_backprop().shape(dim)<<", ";
    } std::cout<<"\b\b]"<<std::endl;
    std::cout<<std::endl;
    
    bool is_pass=true;
    
    int batch = conv.input().shape(0);
    int kernel_h = conv.kernel().shape(0);
    int kernel_w = conv.kernel().shape(1);
    int inchan = conv.kernel().shape(2);
    int outchan = conv.kernel().shape(3);
    int in_h = conv.input().shape(1);
    int in_w = conv.input().shape(2);
    int kernel_shape[4]={kernel_h,kernel_w,inchan,outchan};
    int tensor_shape[4]={batch,in_h,in_w,inchan};
    int input_shape[4]={batch,in_h,in_w,inchan};
    Padding pad_type;
    if(conv.kernel().padding() == "VALID") {
        pad_type = Padding::valid;
        std::cout<<"Padding type VALID"<<std::endl;
    }
    else if(conv.kernel().padding() == "SAME") {
        pad_type = Padding::same;
        std::cout<<"Padding type SAME"<<std::endl;
    }
    else {
        throw "unknown padding type!";
    }
    int stride = conv.kernel().strides(1);
    std::cout<<"Stride "<<stride<<std::endl;
    int out_sizes[2];
    int kernel_sizes[2]={kernel_h, kernel_w};
    shape_inference_with_kernel(out_sizes, input_shape, kernel_sizes, stride, pad_type); 
    int out_h = out_sizes[0];
    int out_w = out_sizes[1];

    // load input
    int input_size=conv.input().data_size();
    if(input_size != batch*inchan*in_h*in_w) {
        throw std::logic_error("Input size does not match its shape!");
    }
    Tensor<int> input(conv.input().data(), conv.input().shape());

    // load kernel
    const int kernel_size=conv.kernel().data_size();
    if(kernel_size != kernel_h*kernel_w*inchan*outchan) {
        throw std::logic_error("Kernel size does not match its shape!");
    }
    Tensor<int> kernel(conv.kernel().data(), conv.kernel().shape()); 

    // load gold output
    const int gold_output_size=conv.output().data_size();
    if(gold_output_size != batch*out_h*out_w*outchan) {
        throw std::logic_error("Gold Output size does not match its shape!");
    }
    Tensor<int> gold_output(conv.output().data(), conv.output().shape());

    // Calc Output
    Tensor<int> output(conv.output().shape());
    conv2d<int>(output.data(), output.shape(), input.data(), input.shape(), kernel.data(), kernel.shape(), stride, pad_type);
    compare_buffers<int>(is_pass, output.data(), gold_output.data(), gold_output.get_size(), "Conv2d forward pass");

    // Calc Filter Grad
    Tensor<int> gold_grad(conv.grad_filter().data(), conv.grad_filter().shape());
    output.reshape(gold_grad);
    conv2d_filter_gradient<int>(output.data(), gold_output.data(), input.data(),
            output.shape(), gold_output.shape(), input.shape(),
            stride, pad_type);
    compare_buffers<int>(is_pass, output.data(), gold_grad.data(), gold_grad.get_size(), "Conv2d filter gradient");

    // Calc Backprop
    gold_grad.load(conv.grad_backprop().data(), conv.grad_backprop().shape());
    output.reshape(gold_grad);
    conv2d_backprop_gradient<int>(output.data(), gold_output.data(), kernel.data(),
            output.shape(), gold_output.shape(), kernel.shape(),
            stride, pad_type);
    compare_buffers<int>(is_pass, output.data(), gold_grad.data(), gold_grad.get_size(), "Conv2d backprop gradient");

    //output.print(); 
    //kernel.print();  
    //gold_grad.print();
  
    print_pass(is_pass);
      
    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
