#ifndef MSE_LOSS_H
#define MSE_LOSS_H

// in_shape = [batch, data]
template<typename data_t>
void mse_loss(float* loss, data_t* in, data_t* labels, int* shape) {
    int batch = shape[0];
    int width = shape[1];
    int size = batch * width;

    float sum=0;
    for(int b=0; b<batch; ++b){
        for(int idx=b*width; idx<(b+1)*width; ++idx) {
            float diff = in[idx] - labels[idx];
            sum += (diff*diff);
        }
    } // batch 
    sum /= (float)size;
    *loss=sum;
}

// in_shape = [batch, data]
template<typename data_t>
void mse_loss_backprop(float* out_grad, data_t* in, data_t* labels, int* shape) {
    int batch = shape[0];
    int width = shape[1];
    int size = batch * width;

    for(int b=0; b<batch; ++b){
        for(int i=0; i<width; ++i) {
            int idx = b*width + i;
            out_grad[idx] = 2*(in[idx] - labels[idx])/size;
        }
    } // batch 
}

#endif //MSE_LOSS_H
