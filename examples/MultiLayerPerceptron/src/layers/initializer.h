#ifndef INITIALIZER_H
#define INITIALIZER_H

#include "tensor.h"
#include <string>

template<typename data_t>
class Initializer{
public:

    void operator()(Tensor<data_t>& target){
        this->apply(target);
    }

    virtual void apply(Tensor<data_t>&) =0; 
};


template<typename data_t>
class ZeroInitializer : public Initializer<data_t>{
public:
    ZeroInitializer<data_t>(){
    }
    virtual void apply(Tensor<data_t>& target) {
        for(int i=0; i<target.get_size(); ++i) {
            target[i] = 0;
        }
    }
};


template<typename data_t>
class NormalTruncInitializer : public Initializer<data_t>{
public:
    NormalTruncInitializer<data_t>(float stddev){
        // random device class instance, source of 'true' randomness for initializing random seed
        std::random_device rd; 
        // Mersenne twister RNG, initialized with seed from previous random device instance
        this->rng=std::mt19937(rd()); 
        //this->rng.seed(42); // constant seed
        this->stddev = stddev;
    }

    virtual void apply(Tensor<data_t>& target) {
        std::normal_distribution<data_t> init(0.0,this->stddev);
        data_t value;
        for(int i=0; i<target.get_size(); ++i) {
            //truncate values
            value=init(rng);
            while(abs(value)>2*this->stddev)
                value = init(rng);
            target[i] = value;
        }
    }
private:
    //std::default_random_engine rng;
    std::mt19937 rng; // random number generator 
    float stddev;
};


template<typename data_t>
class HeInitializer : public Initializer<data_t>{
public:
    HeInitializer<data_t>(){
        // random device class instance, source of 'true' randomness for initializing random seed
        std::random_device rd; 
        // Mersenne twister RNG, initialized with seed from previous random device instance
        this->rng=std::mt19937(rd()); 
        //this->rng.seed(42); // constant seed
    }

    virtual void apply(Tensor<data_t>& target) {
        float fan_inout;
        if(target.is_flat()){
            fan_inout = (float)(target[0]+target[1])/2.0;
        }
        else {
            fan_inout = (float)(target[2]+target[3])/2.0;
        }
        float stddev = std::sqrt(fan_inout);
        std::normal_distribution<data_t> init(0.0,stddev);
        data_t value;
        for(int i=0; i<target.get_size(); ++i) {
            //truncate values
            value=init(rng);
            while(abs(value)>2*stddev)
                value = init(rng);
            target[i] = value;
        }
    }
private:
    //std::default_random_engine rng;
    std::mt19937 rng; // random number generator 
};
#endif // INITIALIZER_H
