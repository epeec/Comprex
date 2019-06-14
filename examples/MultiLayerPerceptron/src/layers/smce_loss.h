#ifndef SMCE_LOSS_H
#define SMCE_LOSS_H

#include "LossLayer.h"

static const int softmax_buffer_size=10000;

// calculates loss and backprop gradient
// in_shape = [batch, data]
// data_t needs to allow fractional part (e.g. float)!
// for backprop, this function is not necessary
// labels are one-hot coded
template<typename data_t>
void smce_loss(float* loss, data_t* grad_out, data_t* in, data_t* labels, int* in_shape) {
    int batch = in_shape[0];
    int width = in_shape[1];
    int size = batch*width;

    *loss=0;
    for(int b=0; b<batch; ++b){
        // find max
        float max_value=in[0];
        for(int i=1; i<width; ++i) {
            int idx = b*width + i;
            float candidate=in[idx];
            if(candidate>max_value)
                max_value = candidate;
        }
        // shift inputs and calculate sum of exps
        float exp_sum=0;
        for(int i=0; i<width; ++i) {
            int idx = b*width + i;
            grad_out[idx] = exp(in[idx]-max_value);
            exp_sum += grad_out[idx]; 
        }
        float log_exp_sum = std::log(exp_sum);
        
        // calculate loss and gradient
        //loss[b]=0;
        for(int i=0; i<width; ++i) {
            int idx = b*width + i;
            //loss[b] += labels[idx] * (log_exp_sum-(in[idx]-max_value));
            *loss += labels[idx] * (log_exp_sum-(in[idx]-max_value));
            grad_out[idx] = (grad_out[idx] / exp_sum - labels[idx]);
        }
    } // batch 
    *loss /= batch;
}


// only calculates backprop gradient
// in_shape = [batch, data]
// data_t must be fractional
template<typename data_t>
void smce_loss_backprop(data_t* grad_out, data_t* in, data_t* labels, int* shape) {
    int batch = shape[0];
    int width = shape[1];

    for(int b=0; b<batch; ++b){
        // find max
        float max_value=0;
        for(int i=0; i<width; ++i) {
            int idx = b*width + i;
            data_t candidate=in[idx];
            if(candidate>max_value)
                max_value = candidate;
        }
        // shift inputs and calculate sum of exps
        data_t exp_sum=0;
        for(int i=0; i<width; ++i) {
            int idx = b*width + i;
            grad_out[idx] = exp(in[idx]-max_value);
            exp_sum += grad_out[idx]; 
        }
        
        // calculate smce_loss_gradient
        for(int i=0; i<width; ++i) {
            int idx = b*width + i;
            grad_out[idx] = (grad_out[idx] / exp_sum - labels[idx]);
        }
    }
}

template<typename data_t>
class SMCELoss : public LossLayer<data_t> {
public:
    // ctor
    SMCELoss<data_t>(std::vector<int> input_shape, const std::string& name=""){ 
        this->input_shape = input_shape;
        this->backprop_grad.reshape(input_shape);
        this->name = name;
    }

    // forward propagation
    virtual float forward(Tensor<data_t> input, Tensor<data_t> labels){
        smce_loss(&(this->output), this->backprop_grad.data(), input.data(), labels.data(), input.shape());
        return this->output;
    }

protected:
};
#endif //SMCE_LOSS_H
