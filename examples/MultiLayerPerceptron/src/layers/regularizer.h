#ifndef REGULARIZER_H
#define REGULARIZER_H

#include "tensor.h"
#include<stdlib.h>

/*****************************************
 * Regularizer IF
 ****************************************/
template<typename data_t>
class Regularizer {
public:
    Regularizer<data_t>(){
    }

    // Calculates the regularization term and returns the contribution to the loss function without weight (lambda)
    virtual float loss(Tensor<data_t>&) = 0;

    // Calculates the contribution to the weight update without weight (lambda)
    virtual Tensor<data_t> get_update(Tensor<data_t>&) = 0;

    Tensor<data_t> operator()(Tensor<data_t>& weights){ 
        return update(weights); 
    }
protected:
};

/*****************************************
 * L2-Regularizer
 ****************************************/
template<typename data_t>
class L2_Regularizer : public Regularizer<data_t> {
public:
    virtual float loss(Tensor<data_t>& weights) {
        float sum=0;
        for(int i=0; i<weights.get_size(); ++i){
            sum += (weights.get_data(i)*weights.get_data(i));
        }
        return sum;
    } 

    virtual Tensor<data_t> get_update(Tensor<data_t>& weights) {
        Tensor<data_t> tensor(weights.get_shape());
        for(int i=0; i<weights.get_size(); ++i){
            tensor[i] = 2*weights.get_data(i);
        }
        return tensor;
    }
};

#endif //REGULARIZER_H
