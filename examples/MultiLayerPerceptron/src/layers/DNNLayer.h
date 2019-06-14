#ifndef DNNLAYER_H
#define DNNLAYER_H

#include "tensor.h"
#include "Layer.pb.h"
#include "DNN.pb.h"
#include <exception>
#include <string>
#include <random>
#include "initializer.h"

template<typename data_t>
class DNNLayer {
public:

    DNNLayer<data_t>(std::vector<int> input_shape) : input_shape(input_shape), output_shape(), output(), backprop_grad(input_shape){
        this->name = "";
    }

    // propagates input and produces new tensor as output.
    // Saves Output in "output".
    virtual Tensor<data_t> forward(Tensor<data_t>) = 0;
    Tensor<data_t> operator()(Tensor<data_t> in) {return forward(in);}
    Tensor<data_t> get_output() {return output;}

    // backpropagates gradient and produces new tensor as output
    // Saves filter gradient in "filter_grad".
    // Saves backprop gradient in "backprop_grad".
    virtual Tensor<data_t> backprop(Tensor<data_t>) = 0;
    Tensor<data_t> get_backprop_grad() {return backprop_grad;}

    std::vector<int> get_output_shape() {return output_shape;}

    // check if this layer needs to be updated
    virtual bool has_kernel() {return false;}

    virtual int get_size() {
        return 0;
    }

    std::string output_shape_string(){
        std::string shape_str;
        shape_str = "[";
        for(int i=0; i<output_shape.size(); ++i) {
            shape_str += std::to_string(output_shape[i]);
            shape_str += ", ";
        }
        shape_str += "\b\b]";
        return shape_str;
    }

    std::string get_name() {return name;}

protected:
    std::vector<int> input_shape; // shape of the input tensor
    std::vector<int> output_shape; // shape of the output tensor
    Tensor<data_t> output; // the layers output
    Tensor<data_t> backprop_grad; // gradient with respect to the input

    std::string name; // Name for this layer
};


template<typename data_t>
class DNNLayerWithKernel : public DNNLayer<data_t> {
public:
    DNNLayerWithKernel<data_t>(std::vector<int> input_shape, std::vector<int> kernel_shape, Initializer<data_t>* initializer)
            : DNNLayer<data_t>(input_shape) {
        // make sure all kernels have 4 dimensions
        if(kernel_shape.size()<4){
            for(int i=kernel_shape.size(); i<4; ++i) {
                kernel_shape[i]=1;
            }
        }
        else if(kernel_shape.size()>4) {
            throw std::runtime_error("Creating kernel with more than 4 dims!");
        }
        this->kernel.reshape(kernel_shape);
        this->kernel_grad.reshape(kernel_shape);
        initializer->apply(this->kernel);
    }

    Tensor<data_t> get_kernel_grad() {return kernel_grad;}

    Tensor<data_t> get_kernel() {return kernel;}

    bool has_kernel() {return true;}

    int get_size(){
        int size=1;
        for(int i=0; i<kernel.get_dims(); ++i) {
            size *= kernel.get_shape(i);
        }
        return size;
    }

    void set_kernel(Tensor<data_t> new_kernel){
        this->kernel = new_kernel;
    }

protected:
    Tensor<data_t> kernel; // the filter weights, can be empty
    Tensor<data_t> kernel_grad; // gradient with respect to the kernel, can be empty
    //Initializer<data_t> initializer; // initializer function for this kernel


};
#endif //DNNLAYER_H
