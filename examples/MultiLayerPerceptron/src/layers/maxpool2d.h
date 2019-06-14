#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

// kernel_shape = [1, height, width, 1]
// in_shape = [batch, height, width, channels]
// out_shape = [batch, height, width, channels] 
void maxpool2d_shape(int* out_shape, int* in_shape, int* kernel_shape, int stride, Padding padding=valid) {
    // input shape
    int batch_size = in_shape[0];
    int in_height = in_shape[1];
    int in_width = in_shape[2];
    int in_chan = in_shape[3];
    // kernel shape
    int kernel_height = kernel_shape[1];
    int kernel_width = kernel_shape[2];
    
    // output shape
    int out_sizes[2];
    int kernel_sizes[2]={kernel_height, kernel_width};
    shape_inference_with_kernel(out_sizes, in_shape, kernel_sizes, stride, padding); 
    int out_height = out_sizes[0];
    int out_width = out_sizes[1];

    // write new shape
    out_shape[0]=batch_size;
    out_shape[1]=out_height;
    out_shape[2]=out_width;
    out_shape[3]=in_chan;
}

// in/out_shape = [batch, height, width, channels]
// kernel_shape: [1,height, width ,1]
// out_map is for backpropagation. Has same shape as "out"
template<typename data_t>
void maxpool2d(data_t* out, int* out_shape, int* out_map, data_t* in, int* in_shape, int* kernel_shape, int stride=1, Padding padding=valid) {
    // input shape
    int batch_size = in_shape[0];
    int in_height = in_shape[1];
    int in_width = in_shape[2];
    int in_chan = in_shape[3];
    // kernel shape
    int kernel_height = kernel_shape[1];
    int kernel_width = kernel_shape[2];
    
    int out_height = out_shape[1];
    int out_width = out_shape[2];
    int out_chan = out_shape[3];
 
    for(int batch=0; batch<batch_size; ++batch) {
        for(int out_y=0; out_y<out_height; ++out_y) {
            for(int out_x=0; out_x<out_width; ++out_x) {
                for(int chan=0; chan<out_chan; ++chan) {
                    int idx_out =   chan
                                  + out_x * out_chan
                                  + out_y * out_width * out_chan 
                                  + batch * out_width * out_height * out_chan;
                    // Kernel
                    for(int kernel_y=0; kernel_y<kernel_height; ++kernel_y) {
                        for(int kernel_x=0; kernel_x<kernel_width; ++kernel_x) {
                            int idx_in = chan 
                                       + (out_x*stride + kernel_x) * in_chan
                                       + (out_y*stride + kernel_y) * in_chan * in_width
                                       + batch * in_chan * in_width * in_height;
                            if(kernel_y==0 && kernel_x==0) {
                                out[idx_out] = in[idx_in];
                                out_map[idx_out] = idx_in;
                            }
                            else if(in[idx_in]>out[idx_out]){
                                out[idx_out] =  in[idx_in];
                                out_map[idx_out] = idx_in;
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename data_t>
void maxpool2d_backprop(data_t* out_grad, int* out_shape, int* grad_map, data_t* in_grad, int* in_shape) {
    // input shape
    int batch = in_shape[0];
    int in_height = in_shape[1];
    int in_width = in_shape[2];
    int in_chan = in_shape[3];
    int in_size = batch * in_height * in_width * in_chan;
    
    // output shape
    int out_height = out_shape[1];
    int out_width = out_shape[2];
    int out_chan = out_shape[3];
    int out_size = batch * out_height * out_width * out_chan;

    // zero output gradient
    for(int i=0; i<out_size; ++i) {
        out_grad[i]=0;
    }

    // backprop gradients based on grad_map
    for(int i=0; i<in_size; ++i) {
        out_grad[grad_map[i]] += in_grad[i];
    }
}

template<typename data_t>
class MaxPool2D : public DNNLayer<data_t> {
public:
    // ctor
    MaxPool2D<data_t>(std::vector<int> input_shape, int filter_size, int stride, const std::string& name="") 
            : DNNLayer<data_t>(input_shape), filter_shape{1, filter_size, filter_size, 1} { 
        this->stride = stride;
        this->padding = Padding::valid;
        this->name = name;
        // infer output shape
        //this->filter_shape = {1,filter_size,filter_size,1};
        //this->filter.reshape(kernel_shape,4);
        int new_shape[4];
        maxpool2d_shape(new_shape, input_shape.data(), filter_shape.data(), stride, padding); 
        this->output.reshape(new_shape, 4); // output shape
        this->output_shape = std::vector<int>(new_shape, new_shape+4);
        this->backinfo.reshape(this->output);
    }

    // forward propagation
    virtual Tensor<data_t> forward(Tensor<data_t> input){
        maxpool2d<data_t>(this->output.data(), this->output.shape(), backinfo.data(), input.data(), input.shape(), filter_shape.data(), stride, padding);
        return this->output;
    }

    // backward propagation
    virtual Tensor<data_t> backprop(Tensor<data_t> in_grad){
        maxpool2d_backprop<data_t>(this->backprop_grad.data(), this->backprop_grad.shape(), backinfo.data(), in_grad.data(), in_grad.shape());
        return this->backprop_grad;
    }
protected:
    std::vector<int> filter_shape;
    Tensor<int> backinfo;
    int stride;
    Padding padding;
};
#endif //MAXPOOL2D_H
