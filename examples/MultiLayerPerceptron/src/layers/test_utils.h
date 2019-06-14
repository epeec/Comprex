
#ifndef TEST_UTILS_H
#define TEST_UTILS_H

template<typename data_t>
void print_tensor(data_t *tensor, int* shape){
    int batch = shape[0];
    int output_h = shape[1];
    int output_w = shape[2];
    int channels = shape[3];

    for(int b=0; b<batch; ++b){
        std::cout<<"Batch: "<<b<<std::endl;
        for (int c=0; c<channels; ++c) {
            std::cout<<"Channel: "<<c<<std::endl;
            for(int y=0; y<output_h; ++y) {
                for(int x=0; x<output_w; ++x) {
                    int idx_in = c
                                + x * channels
                                + y * channels * output_w
                                + b * channels * output_w * output_h;
                    std::cout<<tensor[idx_in]<<" ";
                }
                std::cout<<std::endl;
            }
        std::cout<<std::endl;
        }
    }
}


template<typename data_t>
void print_kernel(data_t *kernel, int* shape){
    int height = shape[0];
    int width = shape[1];
    int in_chan = shape[2];
    int out_chan = shape[3];

    for(int h=0; h<height; ++h){
        //std::cout<<": "<<b<<std::endl;
        for (int w=0; w<width; ++w) {
            std::cout<<"[";
            for(int ic=0; ic<in_chan; ++ic) {
                for(int oc=0; oc<out_chan; ++oc) {
                    int idx = ((h * width + w) * in_chan + ic) * out_chan + oc;
                    std::cout<<kernel[idx]<<" ";
                }
                std::cout<<"; ";
            }
            std::cout<<"\b\b\b]";
        }
        std::cout<<std::endl;
    }
}


template<typename data_t>
void print_vector(data_t *vector, int* shape){
    int batch = shape[0];
    int size = shape[1];

    for(int b=0; b<batch; ++b){
        std::cout<<"Batch: "<<b<<std::endl;
        for(int x=0; x<size; ++x) {
            int idx = x + b * size;
                std::cout<<vector[idx]<<" ";
        }
        std::cout<<std::endl;
    }
}

#endif // TEST_UTILS_H