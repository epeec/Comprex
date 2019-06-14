#ifndef RELU_H
#define RELU_H

// in_shape = [batch, height, width, channels]
// in and out buffer can be the same
template<typename data_t>
void relu(data_t* out, data_t* in, int* in_shape) {
    int size=1;
    for(int i=0; i<4; ++i)
        if(in_shape[i]>0)
            size *= in_shape[i];

    for(int i=0; i<size; ++i) {
        out[i] = (in[i]<0) ? 0 : in[i];
    }
}

// out_grad, in_grad and activation have the same shape
template<typename data_t>
void relu_backprop(data_t* out_grad, data_t* in_grad, data_t* activation, int* in_shape) {
    int size=1;
    for(int i=0; i<4; ++i)
        if(in_shape[i]>0)
            size *= in_shape[i];

    for(int i=0; i<size; ++i) {
        out_grad[i] = (activation[i]<0) ? 0 : in_grad[i];
    }
}

template<typename data_t>
class ReLU : public DNNLayer<data_t> {
public:
    // ctor
    ReLU<data_t>(std::vector<int> input_shape, const std::string& name="") 
            : DNNLayer<data_t>(input_shape) { 
        this->name = name;
        this->output.reshape(input_shape); // output has same shape as input
        this->output_shape = input_shape;
        this->backinfo.reshape(input_shape);
    }

    // forward propagation
    virtual Tensor<data_t> forward(Tensor<data_t> input){
        relu<data_t>(this->output.data(), input.data(), input.shape());
        backinfo = input; // save the input for backprop
        return this->output;
    }

    // backward propagation
    virtual Tensor<data_t> backprop(Tensor<data_t> in_grad){
        relu_backprop<data_t>(this->backprop_grad.data(), in_grad.data(), backinfo.data(), backinfo.shape());
        return this->backprop_grad;
    }
protected:
    Tensor<data_t> backinfo;
};

#endif // RELU_H
