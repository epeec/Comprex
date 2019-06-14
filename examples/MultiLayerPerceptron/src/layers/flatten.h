#ifndef FLATTEN_H
#define FLATTEN_H

// calculate shape of flattened tensor. No reordering!
void flatten(int* out_shape, int* in_shape) {
    out_shape[0]=in_shape[0]; // same batch
    out_shape[1]=in_shape[1]*in_shape[2]*in_shape[3]; // flatten spacial dims
    out_shape[2]=1;
    out_shape[3]=1;
}

template<typename data_t>
class Flatten : public DNNLayer<data_t> {
public:
    // ctor
    Flatten<data_t>(std::vector<int> input_shape, const std::string& name="") 
            : DNNLayer<data_t>(input_shape){ 
        this->name = name;
        int new_shape[4];
        flatten(new_shape, input_shape.data());
        this->output_shape = std::vector<int>(new_shape, new_shape+4);
        this->output.reshape(this->output_shape);
    }

    // forward propagation
    virtual Tensor<data_t> forward(Tensor<data_t> input){
        this->output = input;
        this->output.reshape(this->output_shape);
        return this->output;
    }

    // backward propagation
    virtual Tensor<data_t> backprop(Tensor<data_t> in_grad){
        this->backprop_grad = in_grad;
        this->backprop_grad.reshape(this->input_shape);
        return this->backprop_grad;
    }
protected:
};
#endif //FLATTEN_H
