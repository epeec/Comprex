#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

void fully_connected_shape(int* out_shape, int* in_shape, int* kernel_shape) {
    out_shape[0] = in_shape[0];
    out_shape[1] = kernel_shape[1];
}


/*******************************
 * Calculates the matrix product between input tensor and filter.
 * input and output buffer need to be different!
 * [batch,M] X [M,N] = [batch,N]
 *
 * Args:
 *  out     - buffer for result [batch,N]
 *  in      - buffer for input tensor of shape [batch,M]
 *  in_shape - shape of input array
 *  weights - buffer with filter values [M,N]
 *  weights_shape - buffer with filter shape
 *
 */
template<typename data_t>
void fully_connected(data_t* out, data_t* in, int* in_shape, data_t* weights, int* weights_shape) {
    int M= weights_shape[0];
    int N= weights_shape[1];
    int batch= in_shape[0];

    // calculate
    for(int b=0; b<batch; ++b) {
        for(int n=0; n<N; ++n) {
            int idx_out=b*N+n;
            out[idx_out]=0;
            for(int m=0; m<M; ++m) {
                int idx_in=m+b*M;
                int idx_weight=m*N+n;
                out[idx_out] += in[idx_in] * weights[idx_weight];
            }
        }
    }
}

// output shape [batch,M]
template<typename data_t>
void fc_backprop_gradient(data_t* out_grad, data_t* in_grad, int* grad_shape, data_t* weights, int* weights_shape){
    
    int batch = grad_shape[0];
    int M= weights_shape[0];
    int N= weights_shape[1];

    // calculate
    for(int b=0; b<batch; ++b) {
        for(int m=0; m<M; ++m) {
            int idx_out=b*M+m;
            out_grad[idx_out]=0;
            for(int n=0; n<N; ++n) {
                int idx_in=n+b*N;
                int idx_weight=m*N+n;
                out_grad[idx_out] += in_grad[idx_in] * weights[idx_weight];
            }
        }
    }
}

// output shape [M,N,1,1]
template<typename data_t>
void fc_kernel_gradient(data_t* out_update, data_t* in_grad, int* grad_shape, data_t* activations, int* activation_shape){
    
    int batch = grad_shape[0];
    int N = grad_shape[1];

    // act_height = batch
    int M = activation_shape[1];

    // calculate
    for(int m=0; m<M; ++m) {
        for(int n=0; n<N; ++n) {
            int idx_out=m*N+n;
            out_update[idx_out]=0;
            for(int b=0; b<batch; ++b) {
                int idx_act=m+b*M;
                int idx_grad=n+b*N;
                out_update[idx_out] += activations[idx_act] * in_grad[idx_grad];
            }
        }
    }
}

//kernel_shape=[out_channels]
template<typename data_t>
class FullyConnected : public DNNLayerWithKernel<data_t> {
public:
    // ctor
    FullyConnected<data_t>(std::vector<int> input_shape, std::vector<int> kernel_shape, Initializer<data_t>* initializer, const std::string& name="") 
                : DNNLayerWithKernel<data_t>(input_shape, {input_shape[1], kernel_shape[0], 1, 1}, initializer) { 
        if(!Tensor<data_t>(input_shape).is_flat()){
            throw std::runtime_error("Input for fully connected layer "+name+" must be flat!");
        }
        this->name = name;
        // infer output shape
        int new_shape[4]={0,0,1,1};
        fully_connected_shape(new_shape, input_shape.data(), this->kernel.shape()); 
        this->output.reshape(new_shape, 4); // output shape
        this->output_shape = std::vector<int>(new_shape, new_shape+4);
        backinfo.reshape(input_shape);
    }

    // forward propagation
    virtual Tensor<data_t> forward(Tensor<data_t> input){
        if(this->kernel.is_empty()){
            const std::string message = "Kernel of layer "+this->name+" is not loaded!";
            throw std::runtime_error(message);
        }
        fully_connected<data_t>(this->output.data(), input.data(), input.shape(), this->kernel.data(), this->kernel.shape());
        backinfo = input; // save the input for backprop
        return this->output;
    }

    // backward propagation
    virtual Tensor<data_t> backprop(Tensor<data_t> in_grad){
        fc_backprop_gradient<data_t>(this->backprop_grad.data(), in_grad.data(), in_grad.shape(), this->kernel.data(), this->kernel.shape());
        fc_kernel_gradient<data_t>(this->kernel_grad.data(), in_grad.data(), in_grad.shape(), backinfo.data(), backinfo.shape());
        return this->backprop_grad;
    }
protected:
    Tensor<data_t> backinfo;
};
#endif //FULLY_CONNECTED_H
