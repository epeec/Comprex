#ifndef ARGMAX_H
#define ARGMAX_H

// in_shape: [batch, size]
// out_shape: [batch, 1]
template<typename data_t>
void argmax(data_t* out, data_t* in, int* shape){
    int batch = shape[0];
    int size = shape[1];

    // calculate
    for(int b=0; b<batch; ++b){
        int max_index=0;
        int max_val=in[b*size];
        for(int i=1; i<size; ++i) { //starting from 1, since value of index 0 in max_val
            if(in[i+b*size]>max_val){
                max_index=i;
                max_val=in[i+b*size];
            }
        }
        out[b]=max_index;
    }
    
}

template<typename data_t>
class Argmax : public DNNLayer<data_t> {
public:
    // ctor
    Argmax<data_t>(std::vector<int> input_shape, const std::string& name="") 
        : DNNLayer<data_t>(input_shape){ 
        this->name = name;
        int new_shape[4] = {input_shape[0], 1, 1, 1}; // only batch_dimension remains
        this->output.reshape(new_shape, 4);
        this->output_shape = std::vector<int>(new_shape, new_shape+4);
    }

    // forward propagation
    virtual Tensor<data_t> forward(Tensor<data_t> input){
        argmax(this->output.data(), input.data(), input.shape());
        return this->output;
    }

    // backward propagation
    virtual Tensor<data_t> backprop(Tensor<data_t> in_grad){
        throw std::runtime_error("Argmax is not supposed to be backpropagated!");
        return this->backprop_grad;
    }
protected:
};
#endif //ARGMAX_H
