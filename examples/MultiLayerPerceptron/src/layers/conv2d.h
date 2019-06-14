#ifndef CONV2D_H
#define CONV2D_H

#include "layers.h"

/************************
 * Calculate the output shape of a convolution.
 * Args:
 *  out_shape: resulting shape after convolution.
 *  in_shape: shape of input tensor
 *  kernel_shape: shape of the convolution kernel
 *  stride: striding factor for convolution.
 *  padding: determines height and width of output tensor.
 *
 * input/output - shape: [batch, height, width, channels]
 * kernel - shape: [height, width, inchan, outchan]
 */
void conv2d_shape(int* out_shape, int* in_shape, int* kernel_shape, int stride, Padding padding) {
    // input shape
    int batch_size = in_shape[0];
    int in_height = in_shape[1];
    int in_width = in_shape[2];
    int in_chan = in_shape[3];
    // kernel shape
    int kernel_height = kernel_shape[0];
    int kernel_width = kernel_shape[1];
    int kernel_inchan = kernel_shape[2];
    int kernel_outchan = kernel_shape[3];
    
    // output shape
    int out_sizes[2];
    int kernel_sizes[2]={kernel_height, kernel_width};
    shape_inference_with_kernel(out_sizes, in_shape, kernel_sizes, stride, padding); 
    int out_height = out_sizes[0];
    int out_width = out_sizes[1];
    int out_chan = kernel_outchan;

    // write new shape
    out_shape[0]=batch_size;
    out_shape[1]=out_height;
    out_shape[2]=out_width;
    out_shape[3]=out_chan;
}

/***************************************
 * Convolve input with kernel.
 * Result will be written to 'out'. Make sure 'in' is not pointing to 'out'!
 * output shape 'out_shape' needs to be determined in advance.
 * Args:
 *  out     - buffer where result will be written
 *  in      - input tensor
 *  kernel  - filter data
 *  out_shape - shape of output tensor. Needs to be calculated in advance.
 *  in_shape - shape of input tensor
 *  kernel_shape - shape of convolution kernel
 *  stride  - striding step size
 *  padding - input padding, determines output shape
 *
 * input/output - shape: [batch, height, width, channels]
 * kernel - shape: [height, width, inchan, outchan]
 */
template<typename data_t>
void conv2d(data_t* out, int* out_shape, data_t* in, int* in_shape, data_t* kernel, int* kernel_shape, int stride=1, Padding padding=valid) {
    // input shape
    int batch_size = in_shape[0];
    int in_height = in_shape[1];
    int in_width = in_shape[2];
    int in_chan = in_shape[3];
    // kernel shape
    int kernel_height = kernel_shape[0];
    int kernel_width = kernel_shape[1];
    int kernel_inchan = kernel_shape[2];
    int kernel_outchan = kernel_shape[3];
    
    // padding
    int paddings[4]; 
    infer_padding(paddings, in_shape, kernel_shape, stride, padding);
    int pad_top = paddings[0];
    int pad_bottom = paddings[1];
    int pad_left= paddings[2];
    int pad_right= paddings[3];
    //std::cout<<"Padding: "<<pad_top<<", "<<pad_bottom<<", "<<pad_left<<", "<<pad_right<<std::endl;

    // output shape
    // conv2d_shape(out_shape, in_shape, kernel_shape, stride, padding);
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
                    out[idx_out] = 0;
                    // Kernel
                    for(int kernel_y=0; kernel_y<kernel_height; ++kernel_y) {
                        for(int kernel_x=0; kernel_x<kernel_width; ++kernel_x) {
                            for(int kernel_in=0; kernel_in<kernel_inchan; ++kernel_in) {
                                int in_x = out_x*stride+kernel_x-pad_left;
                                int in_y = out_y*stride+kernel_y-pad_top;
                                if(in_x>=0 && in_x<in_width 
                                && in_y>=0 && in_y<in_height) {
                                    int idx_kernel = chan
                                                   + kernel_in * kernel_outchan
                                                   + kernel_x * kernel_inchan * kernel_outchan
                                                   + kernel_y * kernel_width * kernel_inchan * kernel_outchan;
                                    int idx_in = kernel_in 
                                               + in_x  * in_chan
                                               + in_y  * in_chan * in_width
                                               + batch * in_chan * in_width * in_height;
                                    out[idx_out] += in[idx_in] * kernel[idx_kernel];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename data_t>
void conv2d_kernel_gradient(data_t* out_update, data_t* in_grad, data_t* activation, 
        int* update_shape, int* grad_shape, int* act_shape, 
        int stride=1, Padding padding=valid) {

    // input shape
    int batch = grad_shape[0];
    int grad_height = grad_shape[1];
    int grad_width = grad_shape[2];
    int grad_chan = grad_shape[3];

    // activation shape
    int act_height = act_shape[1];
    int act_width = act_shape[2];
    int act_chan = act_shape[3];

    // kernel shape
    int kernel_height = update_shape[0];
    int kernel_width = update_shape[1];
    int kernel_inchan = update_shape[2];
    int kernel_outchan = update_shape[3];
    
    // padding
    int paddings[4]; 
    infer_padding(paddings, act_shape, update_shape, stride, padding);
    int pad_top = paddings[0];
    int pad_bottom = paddings[1];
    int pad_left= paddings[2];
    int pad_right= paddings[3];
 
    for(int k_y=0; k_y<kernel_height; ++k_y) {
        for(int k_x=0; k_x<kernel_width; ++k_x) {
            for(int a_chan=0; a_chan<act_chan; ++a_chan) {
                for(int g_chan=0; g_chan<grad_chan; ++g_chan) {
                    int idx_kernel = ((k_y * kernel_width + k_x) * act_chan + a_chan ) * grad_chan + g_chan;
                    out_update[idx_kernel] = 0;
                    // Kernel
                    for(int b=0; b<batch; ++b) {
                        for(int g_y=0; g_y<grad_height; ++g_y) {
                            for(int g_x=0; g_x<grad_width; ++g_x) {
                                int act_x = k_x+g_x*stride-pad_left;
                                int act_y = k_y+g_y*stride-pad_top;
                                if(act_x>=0 && act_x<act_width 
                                && act_y>=0 && act_y<act_height) {
                                    int idx_grad = ((b*grad_height + g_y)*grad_width + g_x )*grad_chan+g_chan;
                                    int idx_act = ((b*act_height + act_y)*act_width + act_x )*act_chan+a_chan;
                                    out_update[idx_kernel] += activation[idx_act] * in_grad[idx_grad];
                                }
                            }
                        }
                    }
                    //std::cout<<"["<<k_y<<","<<k_x<<","<<a_chan<<","<<g_chan<<"] = "<<idx_kernel<<" "<<out_update[idx_kernel]<<std::endl;
                }
            }
        }
    }
}


template<typename data_t>
void conv2d_backprop_gradient(data_t* out_grad, data_t* in_grad, data_t* weights, 
        int* out_grad_shape, int* in_grad_shape, int* weights_shape, 
        int stride=1, Padding padding=valid) {

    // input shape
    int batch = in_grad_shape[0];
    int grad_height = in_grad_shape[1];
    int grad_width = in_grad_shape[2];
    int grad_chan = in_grad_shape[3];

    // kernel shape
    int kernel_height = weights_shape[0];
    int kernel_width = weights_shape[1];
    int kernel_inchan = weights_shape[2];
    int kernel_outchan = weights_shape[3];
    
    // output shape;
    int out_height = out_grad_shape[1];
    int out_width = out_grad_shape[2];
    int out_chan = out_grad_shape[3];

    // padding
    int paddings[4]; 
    infer_padding(paddings, out_grad_shape, weights_shape, stride, padding);
    //TODO: is this padding rule general enough?
    int pad_top = kernel_height - 1 - paddings[0]; 
    int pad_bottom = kernel_height - 1 - paddings[1];
    int pad_left= kernel_width - 1 - paddings[2]; 
    int pad_right= kernel_width - 1 - paddings[3]; 
    //std::cout<<"Padding: "<<pad_top<<", "<<pad_bottom<<", "<<pad_left<<", "<<pad_right<<std::endl;
 
    int out_x_stride=0;
    int out_y_stride=0;
    int out_x_count=0;
    int out_y_count=0;

    int pad_top_stride = pad_top/stride;
    int pad_left_stride = pad_left/stride;
    int pad_top_mod = pad_top - pad_top_stride*stride;
    int pad_left_mod = pad_left - pad_left_stride*stride;

    for(int b=0; b<batch; ++b) {
        for(int out_y=0; out_y<out_height; ++out_y) {
            for(int out_x=0; out_x<out_width; ++out_x) {
                for(int out_c=0; out_c<out_chan; ++out_c) {
                    int idx_out = ((b * out_height + out_y) * out_width + out_x ) * out_chan + out_c;
                    out_grad[idx_out] = 0;
                    //std::cout<<"out["<<b<<","<<out_y<<","<<out_x<<","<<out_c<<"]=";
                    if(out_x==0){
                        out_x_count=0;
                        out_x_stride=0; // index correction for gradient this is out_x%stride
                    }

                    if(out_y==0) {
                        out_y_count=0;
                        out_y_stride=0;
                    }
                    if(out_x_stride==stride){
                        ++out_x_count;
                        out_x_stride=0;
                    }
                    if(out_y_stride==stride) {
                        ++out_y_count;
                        out_y_stride=0;
                    }

                    // Kernel
                    for(int g_y=0; g_y<kernel_height; ++g_y) { // due to striding, receptive field is at least kernel-sized
                        for(int g_x=0; g_x<kernel_width; ++g_x) {
                            for(int g_c=0; g_c<grad_chan; ++g_c) {
                                int g_xx= out_x_count+g_x-pad_left_stride; // for gradient index, effectively, this is out_x/stride+g_x
                                int g_yy= out_y_count+g_y-pad_top_stride;
                                int k_x= -out_x_stride+g_x*stride+pad_left_mod; // the gradient coordinates in dilated space
                                int k_y= -out_y_stride+g_y*stride+pad_top_mod;
                                //std::cout<<g_xx<<std::endl;
                                if( k_x>=0 && k_x<kernel_width
                                 && k_y>=0 && k_y<kernel_height
                                 && g_xx>=0 && g_yy>=0 
                                 && g_xx<grad_width && g_yy<grad_height) {
                                    int k_yy = kernel_height-1-k_y; // rotate filter 180 degree
                                    int k_xx = kernel_width-1-k_x;
                                    int idx_grad = ((b*grad_height + g_yy)*grad_width + g_xx )*grad_chan+g_c;
                                    int idx_weights = ((k_yy * kernel_width + k_xx) * out_chan + out_c) * grad_chan + g_c;
                                    out_grad[idx_out] += in_grad[idx_grad] * weights[idx_weights];
                                    //std::cout<<"in["<<b<<","<<g_yy<<","<<g_xx<<","<<g_c<<"]"<< \
                                            " * weight["<<k_yy<<","<<k_xx<<","<<out_c<<","<<g_c<<"] +";
                                    //std::cout<<in_grad[idx_grad]<<"*"<<weights[idx_weights]<<"+";
                                }
                            }
                        }
                    }
                    //std::cout<<std::endl;
                    //std::cout<<"\b = "<<out_grad[idx_out]<<std::endl;
                    //std::cout<<"["<<k_y<<","<<k_x<<","<<a_chan<<","<<g_chan<<"] = "<<idx_kernel<<" "<<out_update[idx_kernel]<<std::endl;
                }
                ++out_x_stride;
            }
            ++out_y_stride;
        }
    }
}

// kernel_shape = [height, width, out_channels]
template<typename data_t>
class Conv2D : public DNNLayerWithKernel<data_t> {
public:
    // ctor
    Conv2D<data_t>(std::vector<int> input_shape, std::vector<int> kernel_shape, Initializer<data_t>* initializer, int stride, Padding padding, const std::string& name="") 
            : DNNLayerWithKernel<data_t>(input_shape, {kernel_shape[0], kernel_shape[1], input_shape[3], kernel_shape[2]}, initializer) { 
        this->stride = stride;
        this->padding = padding;
        this->name = name;
        // infer output shape
        int new_shape[4];
        conv2d_shape(new_shape, input_shape.data(), this->kernel.shape(), stride, padding);
        this->output.reshape(new_shape, 4); // output shape
        this->output_shape=std::vector<int>(new_shape, new_shape+4);
        this->backinfo.reshape(input_shape);
    }

    // forward propagation
    virtual Tensor<data_t> forward(Tensor<data_t> input){
        if(this->kernel.is_empty()){
            const std::string message = "Kernel of layer "+this->name+" is not loaded!";
            throw std::runtime_error(message);
        }
        conv2d<data_t>(this->output.data(), this->output.shape(), input.data(), input.shape(), this->kernel.data(), this->kernel.shape(), this->stride, this->padding);
        this->backinfo = input; // save the input for backprop
        return this->output;
    }

    // backward propagation
    virtual Tensor<data_t> backprop(Tensor<data_t> in_grad){
        conv2d_backprop_gradient<data_t>(this->backprop_grad.data(), in_grad.data(), this->kernel.data(), this->backprop_grad.shape(), in_grad.shape(), this->kernel.shape(), this->stride, this->padding);
        conv2d_kernel_gradient<data_t>(this->kernel_grad.data(), in_grad.data(), backinfo.data(), this->kernel_grad.shape(), in_grad.shape(), backinfo.shape(), stride, padding);
        return this->backprop_grad;
    }
protected:
    Tensor<data_t> backinfo; // backprop aux information
    int stride;
    Padding padding;
};
#endif //CONV2D_H
