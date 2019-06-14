#ifndef LOSSLAYER_H
#define LOSSLAYER_H

#include "tensor.h"
#include <exception>
#include <string>

template<typename data_t>
class LossLayer {
public:

    LossLayer<data_t>() : input_shape(), backprop_grad(){
        this->output = -1;
        this->name = "";
    }

    // combines input with labels and produces single scalar as output.
    // Saves Output in "output".
    // Saves Gradient in "backprop_grad"
    virtual float forward(Tensor<data_t>, Tensor<data_t>) = 0;
    Tensor<data_t> operator()(Tensor<data_t> in) {return forward(in);}

    float get_output() {return output;}
    Tensor<data_t> get_backprop_grad() {return backprop_grad;} 

    std::string get_name() {return name;}

protected:
    std::vector<int> input_shape; // shape of the input tensor
    float output; // the layers output
    Tensor<data_t> backprop_grad; // gradient with respect to the input

    std::string name; // Name for this layer
};
#endif //LOSSLAYER_H
