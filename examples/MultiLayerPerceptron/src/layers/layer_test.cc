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


{
    std::cout<<"Fully Connected"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer fc;
    const std::string filename = test_data_folder + "fc.pb";
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
   
} // Fully Connected

{
    std::cout<<"Convolution"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer conv;
    const std::string filename = test_data_folder + "conv_valid_ns.pb";
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

    //gold_output.print();
    //kernel.print();
    //gold_grad.print();

    print_pass(is_pass);
    
} // Convolution

{
    std::cout<<"ReLU"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer relu_layer;
    const std::string filename = test_data_folder + "relu.pb";
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
} // Relu

// Max Pool
{
    std::cout<<"Max Pool"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer maxpool_layer;
    const std::string filename = test_data_folder + "maxpool_VALID.pb";
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
} // Max_pooling
    
    
{
    std::cout<<"Flatten"<<std::endl;
    std::cout<<"############################"<<std::endl;

    bool is_pass=true;
    int error_count=0;

    int shape[4] = {32,64,64,128};
    int gold_shape[4] = {shape[0],shape[1]*shape[2]*shape[3],1,1};
    flatten(shape, shape);
    
    compare_buffers<int>(is_pass, shape, gold_shape, 4, "Flatten");

    print_pass(is_pass);
} // Flatten
 
{
    std::cout<<"Add Bias"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer bias_layer;
    const std::string filename = test_data_folder + "add_bias.pb";
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
}

{
    std::cout<<"Softmax"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer softmax_layer;
    const std::string filename = test_data_folder + "softmax.pb";
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

} // Softmax


{
    std::cout<<"MSE"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer mse_layer;
    const std::string filename = test_data_folder + "mse.pb";
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
    
} // MSE

{
    std::cout<<"Softmax-Cross-Entropy Loss"<<std::endl;
    std::cout<<"############################"<<std::endl;
    Layer::layer smce_layer;
    const std::string filename = test_data_folder + "smce.pb";
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
} //SMCE

{
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
}

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
