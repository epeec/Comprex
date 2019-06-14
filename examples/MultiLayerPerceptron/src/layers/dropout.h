#ifndef DROPOUT_H
#define DROPOUT_H

#include <cstdlib>

// keep_prob= a float between 0.0 and 1.0
template<typename data_t>
void dropout(data_t* out, int* map, data_t* in, int* shape, float keep_prob) {
    int batch = shape[0];
    int width = shape[1];
    int height = shape[2];
    int chan = shape[3];
    int size = batch * width * height * chan;

    // check if keep_prob is between 1.0 and 0.0
    assert(keep_prob>=0 && keep_prob<=1);

    for(int i=0; i<size; ++i){
        float random = (float)std::rand()/(float)RAND_MAX;
        if(random<=keep_prob){
            out[i]=in[i]/keep_prob;
            map[i]=1;
        }
        else{
            out[i]=0;
            map[i]=0;
        }
    }
}

template<typename data_t>
void dropout_backprop(data_t* grad_out, int* map, data_t* grad_in, int* shape, float keep_prob) {
    int batch = shape[0];
    int width = shape[1];
    int height = shape[2];
    int chan = shape[3];
    int size = batch * width * height * chan;

    // check if keep_prob is between 1.0 and 0.0
    assert(keep_prob>=0 && keep_prob<=1);

    for(int i=0; i<size; ++i){
        grad_out[i] = grad_in[i]*(float)map[i]/keep_prob;
    }
}

template<typename data_t>
class Dropout : public DNNLayer<data_t> {
public:
    // ctor
    Dropout<data_t>(std::vector<int> input_shape, float keep_prob, const std::string& name="") 
            : DNNLayer<data_t>(input_shape){ 
        this->keep_prob = keep_prob;
        this->training = false;
        this->name = name;
        // infer output shape
        this->output_shape = input_shape;
        this->output.reshape(input_shape); // output reshape
        this->backinfo.reshape(input_shape);
    }

    // forward propagation
    virtual Tensor<data_t> forward(Tensor<data_t> input){
        if(training){
            dropout<data_t>(this->output.data(), backinfo.data(), input.data(), input.shape(), keep_prob);
        }
        else {
            this->output=input;
        }
        return this->output;
    }

    // backward propagation
    virtual Tensor<data_t> backprop(Tensor<data_t> in_grad){
        if(training) {
            dropout_backprop<data_t>(this->backprop_grad.data(), backinfo.data(), in_grad.data(), in_grad.shape(), keep_prob);
        }
        else {
            this->backprop_grad = in_grad;
        }
        return this->backprop_grad;
    }

    void enable_training(){
        training=true;
    }

    void disable_training(){
        training=false;
    }

    bool is_training() {
        return training;
    }

protected:
    Tensor<int> backinfo;
    float keep_prob;
    bool training;
};
#endif //DROPOUT_H
