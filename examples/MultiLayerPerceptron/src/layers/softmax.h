#ifndef SOFTMAX_H
#define SOFTMAX_H

// in_shape = [batch, data]
// data_t needs to allow fractional part (e.g. float)!
template<typename data_t>
void softmax(data_t* out, data_t* in, int* in_shape) {
    int batch = in_shape[0];
    int size = in_shape[1];

    for(int b=0; b<batch; ++b){
        // find max
        float max_value=0;
        for(int idx=b*size; idx<(b+1)*size; ++idx) {
            float candidate=in[idx];
            if(candidate>max_value)
                max_value = candidate;
        }
        // shift inputs and calculate sum of exps
        float exp_sum=0;
        for(int idx=b*size; idx<(b+1)*size; ++idx) {
            out[idx] = exp((float)in[idx]-max_value);
            exp_sum += out[idx]; 
        }
        
        // calculate softmax
        for(int idx=b*size; idx<(b+1)*size; ++idx) {
            out[idx] = out[idx] / exp_sum;
        }
    } // batch 
}


// in_shape = [batch, data]
// data_t needs to have fractional part (e.g. float)
template<typename data_t>
void softmax_backprop(data_t* grad_out, data_t* grad_in, data_t* sm_in, int* shape) {
    int batch = shape[0];
    int width = shape[1];

    for(int b=0; b<batch; ++b){
        for(int out_w=0; out_w<width; ++out_w) { 
            int idx_out= b*width + out_w;
            grad_out[idx_out]=0;
            for(int i=0; i<width; ++i) {
                int idx_in = b*width + i;
                data_t kron_d = (out_w==i) ? 1 : 0;
                grad_out[idx_out] += sm_in[idx_in] * (kron_d-sm_in[idx_out])*grad_in[idx_in];
            }
        }
    } // batch 
}
#endif //SOFTMAX_H
