#ifndef ADD_BIAS_H
#define ADD_BIAS_H

#include "DNNLayer.h"
#include <exception>

/* Adds bias to a tensor
 * Non-flat tensors add bias differently than flat ones, set flag is_flat if tensor is flat!
 * Args:
 *      out: output buffer, where results are written to.
 *      in: input buffer, one of the summands.
 *      bias: bias buffer, the other summand. One entry per channel.
 *      shape: shape of the input and output. [batch, height, width, channels]
 *      is_flat: if the input is flat (shape-dims=2), the bias is added differently than if shape-dims=4
 */
template<typename data_t>
void add_bias(data_t* out, data_t* in, data_t* bias, int* shape, bool is_flat=false) {
    int batch, size, rows;
    batch = shape[0];
    if(is_flat){
        size=1;
        rows=shape[1];
    }
    else {
        size = shape[1] * shape[2];
        rows = shape[3];
    }
    for(int b=0; b<batch; ++b) {
        for(int i=0; i<size; ++i) {
            for(int r=0; r<rows; ++r) {
                int idx = r
                        + i * rows
                        + b * rows * size;
                out[idx] = in[idx] + bias[r];
            }
        }
    }
}

// gradient backprop, just copy gradient to output
template<typename data_t>
void add_bias_backprop_grad(data_t* out, data_t* in_grad, int* shape, bool is_flat=false){
    int size;
    if(is_flat) {
        size = shape[0] * shape[1];
    }
    else {
        size = shape[0] * shape[1] * shape[2] * shape[3];
    }
    for(int i=0; i<size; ++i) {
        out[i] = in_grad[i];
    }
}

// for every last dim (i.e. width or channels), add up all elements
// output of shape [rows,1,1,1]
template<typename data_t>
void add_bias_kernel_grad(data_t* out, data_t* in_grad, int* shape, bool is_flat=false){
    int size, rows;
    if(is_flat){
        size=shape[0];
        rows=shape[1];
    }
    else {
        size = shape[0] * shape[1] * shape[2];
        rows = shape[3];
    }

    for(int r=0; r<rows; ++r) {
        out[r]=0;
        for(int i=0; i<size; ++i) {
            int idx = i*rows+r;
            out[r] += in_grad[idx];
        }
    }
}

/* Class implementing a bias layer
 *
 *
 */
template<typename data_t>
class AddBias : public DNNLayerWithKernel<data_t> {
public: 
    // ctor
    AddBias<data_t>(std::vector<int> input_shape, Initializer<data_t>* initializer, const std::string& name="")
        : DNNLayerWithKernel<data_t>(input_shape, {1,1,1,1}, initializer) { 
        this->name = name;
        this->output_shape = input_shape;
        this->output.reshape(input_shape); // output shape is the same as input
        int n_biases;
        if(Tensor<data_t>(input_shape).is_flat()){
            n_biases=input_shape[1];
        }
        else {
            n_biases = input_shape[3];
        }
        this->kernel.reshape({n_biases,1,1,1});
        this->kernel_grad.reshape(this->kernel);
        initializer->apply(this->kernel);
    }

    // forward propagation
    virtual Tensor<data_t> forward(Tensor<data_t> input){
        if(this->kernel.is_empty()){
            const std::string message = "Kernel of layer "+this->name+" is not loaded!";
            throw std::runtime_error(message);
        }
        add_bias<data_t>(this->output.data(), input.data(), this->kernel.data(), input.shape(), input.is_flat());
        return this->output;
    }

    // backward propagation
    virtual Tensor<data_t> backprop(Tensor<data_t> in_grad){
        add_bias_backprop_grad<data_t>(this->backprop_grad.data(), in_grad.data(), this->backprop_grad.shape(), in_grad.is_flat());
        add_bias_kernel_grad<data_t>(this->kernel_grad.data(), in_grad.data(), this->kernel_grad.shape(), in_grad.is_flat());
        return this->backprop_grad;
    }
protected:

};

#endif //ADD_BIAS_H
