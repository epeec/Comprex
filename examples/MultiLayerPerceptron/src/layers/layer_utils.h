#ifndef LAYER_UTILS_H
#define LAYER_UTILS_H

#include <math.h>

enum Padding { valid, same };

// in/out_shape: [batch, height, width, channels]
// kernel_shape: [height, width]
// out_sizes: [new_height, new_width]
void shape_inference_with_kernel(int* out_sizes, const int* in_shape, const int* kernel_shape, const int stride=1, const Padding padding=valid){ 
    int out_height;
    int out_width;
    if(padding == same) {
        out_height = ceil((float)(in_shape[1])/(float)stride);
        out_width =  ceil((float)(in_shape[2])/(float)stride);
    }
    else if(padding == valid) {
        out_height = ceil((float)(in_shape[1] - kernel_shape[0]+1)/(float)stride);
        out_width =  ceil((float)(in_shape[2] - kernel_shape[1]+1)/(float)stride);
    }
    else {
        throw "padding type unknown.";
    }
    out_sizes[0] = out_height;
    out_sizes[1] = out_width;
}

// calculates padding size based on filter dimension and stride dimension
// https://www.tensorflow.org/api_guides/python/nn
int calc_pad(int inp, int filt, int stride){
    return std::max(0, filt - ((inp+stride-1)%stride +1) );
}

// input_shape: [batch, height, width, channels]
// filter_shape: [height, width, x, x]
// out: [top, bottom, left, right]
void infer_padding(int* out, int* input_shape, int* filter_shape, int stride, Padding padding){
    /*
    Optimal padding according to Tensorflow implementation.
    https://www.tensorflow.org/api_guides/python/nn
    */
    int padding_h;
    int padding_w; 

    int strides[2]={stride,stride};

    if(padding == valid) {
        padding_h = 0;
        padding_w = 0;
    }
    else if( padding == same){
        padding_h = calc_pad(input_shape[1], filter_shape[0], strides[0]);
        padding_w = calc_pad(input_shape[2], filter_shape[1], strides[1]);
    }
    else {
        throw "Padding type not supported.";
    }

    // top
    out[0]=padding_h/2; // integer division
    // bottom
    out[1]=padding_h-out[0];
    // left
    out[2]=padding_w/2; // integer division
    // right
    out[3]=padding_w-out[2]; 
}
#endif //LAYER_UTILS_H
