#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include<vector>
#include<string>
#include<algorithm>
#include "regularizer.h"
#include "comprex.hxx"

template<typename data_t>
class Comprex_Optimizer {
public:
    //types
    typedef std::vector<DNNLayer<data_t>*> layer_list_t;

    // ctor
    Comprex_Optimizer<data_t>(gaspi::Runtime& gaspi_runtime, gaspi::Context& gaspi_context)
                : gaspi_myRank(gaspi_context.rank()),
                  gaspi_chiefRank(0)
                  {
        this->layers = layer_list_t(0);
        this->regularizer = NULL;
        this->weight_decay = 0.0;
        // init GaspiCxx
        this->gaspi_runtime = &gaspi_runtime;
        this->gaspi_context = &gaspi_context;
        this->gaspi_segment = new gaspi::segment::Segment(static_cast<size_t>(1) << 28);
        //this->gaspi_myRank =  gaspi_context.rank();
        this->gaspi_myRank_int = static_cast<int> ( gaspi_context.rank().get() );
        this->gaspi_num_ranks = gaspi_context.size().get();
        this->gaspi_com_tag = 1;
        this->comprex_threshold = 0.1;
        this->total_weights_size=0;
        this->comprex_handle = new compressed_exchange::ComprExRunLengths<data_t>(gaspi_runtime, gaspi_context, *gaspi_segment, total_weights_size);
        
        this->cmprx_number_of_runs=0;
        this->cmprx_accumulated_sparsity=0;
    }

    ~Comprex_Optimizer<data_t>(){
        if(this->regularizer != NULL) {
            delete this->regularizer;
            delete this->comprex_handle;
            delete this->gaspi_segment;
        }
    }

    void set_layers(layer_list_t layers){
        this->layers=layers;
        // set new size for comprex
        // estimate new size
        total_weights_size=0;
        for(typename layer_list_t::iterator it=layers.begin(); it != layers.end(); ++it) {
            total_weights_size += (*it)->get_size();
        }
        delete this->comprex_handle;
        this->comprex_handle = new compressed_exchange::ComprExRunLengths<data_t>(*gaspi_runtime, *gaspi_context, *gaspi_segment, total_weights_size);
    }

    void set_regularizer(std::string regularizer_type, float weight_decay){
        if(this->regularizer != NULL){
            delete this->regularizer;
        }
        if(regularizer_type == "L2"){
            this->regularizer = new L2_Regularizer<data_t>();
        }
        else {
            throw std::runtime_error("Regularizer \""+regularizer_type+"\" is not supported!");
        }
        this->weight_decay=weight_decay;

    }

    // applies update to weights
    void apply(){
        if(layers.size()==0){
            std::cout<<"Warning: Optimizer has no layers to optimize!"<<std::endl;
        }
        // gradients blob for reduction
        //std::vector<data_t> gradients_blob(total_weights_size);
        std::unique_ptr<data_t[]> gradients_blob(new data_t[total_weights_size]);
        std::unique_ptr<data_t[]> remote_gradients_blob(new data_t[total_weights_size]);
        int blob_offset=0;
        // update blob
        //std::vector<data_t> update_blob(total_weights_size);

        // transform layer gradients into blob
        for(typename layer_list_t::iterator it=layers.begin(); it != layers.end(); ++it) {
            if(!(*it)->has_kernel()){
                continue;
            }
            Tensor<data_t> gradient(dynamic_cast<DNNLayerWithKernel<data_t>*>(*it)->get_kernel_grad());
            for(int i=0; i<gradient.get_size();++i){
                gradients_blob[i+blob_offset] = gradient[i];
            }
            blob_offset += gradient.get_size();
        }

        // REDUCTION
        for(int it=1; it < gaspi_num_ranks; it*=2){
            int destRank = gaspi_myRank_int - it;
            int srcRank = gaspi_myRank_int + it;
            // Receive right
            if(gaspi_myRank_int%(it*2) == 0){
                // check if source exists
                if (srcRank < gaspi_num_ranks) {
                    comprex_handle->p2pVectorGetRemote(
        	                  remote_gradients_blob, static_cast<gaspi::group::Rank>(srcRank), gaspi_com_tag);
                    // local reduce
                    for(int i=0; i<total_weights_size;++i){
                        gradients_blob[i] = gradients_blob[i] + remote_gradients_blob[i];
                    }
                }

            }
            //Send left
            else if(gaspi_myRank_int%(it*2) == it){
                // check if target exists
                if (destRank >= 0){
                    comprex_handle->compress_and_p2pVectorWriteRemote(
        	                  gradients_blob, comprex_threshold, static_cast<gaspi::group::Rank>(destRank), gaspi_com_tag);
                    // make sparsity statistics
                    const data_t* pRestsV = comprex_handle->entryPointerRestsVector();
                    int zeros=0;
                    for(int i=0; i<total_weights_size;++i){
                      if(pRestsV[i]==0) ++zeros;
                    }
                    double sparsity = 1-(float)zeros/(float)total_weights_size;
                    printf("Rank %d Transmission sparsity=%f\n", gaspi_myRank.get(), sparsity);
                    ++cmprx_number_of_runs;
                    cmprx_accumulated_sparsity += sparsity;
                }
            }
            gaspi_context->barrier();
        }

        // BROADCAST
        // broadcast new model parameters
        if(gaspi_myRank == gaspi_chiefRank){
            // send update
            for(int rank=1; rank < gaspi_num_ranks; ++rank) {
                comprex_handle->compress_and_p2pVectorWriteRemote(
    	                  gradients_blob, 0.0, static_cast<gaspi::group::Rank>(rank), gaspi_com_tag);
            }
        }
        if(gaspi_myRank != gaspi_chiefRank){
            // receive update
            comprex_handle->p2pVectorGetRemote(
	                  gradients_blob, gaspi_chiefRank, gaspi_com_tag);
        }
        gaspi_context->barrier();

        // transform gradient blob back to layer and apply update
        blob_offset=0;
        for(typename layer_list_t::iterator it=layers.begin(); it != layers.end(); ++it) {
            if(!(*it)->has_kernel()){
                continue;
            }
            Tensor<data_t> kernel(dynamic_cast<DNNLayerWithKernel<data_t>*>(*it)->get_kernel());
            Tensor<data_t> gradient(kernel.get_shape());
            gradient.load(gradients_blob.get()+blob_offset);
            Tensor<data_t> grad_update = this->get_update(&gradient);
            kernel = kernel - grad_update;
            // apply weight decay
            if(regularizer != NULL) {
                //std::cout<<"Regularizer Loss: "<<regularizer->loss(new_kernel)<<std::endl;
                kernel = kernel - (regularizer->get_update(kernel) * weight_decay);
            }
            dynamic_cast<DNNLayerWithKernel<data_t>*>(*it)->set_kernel(kernel); //update
            blob_offset += kernel.get_size();
        }

            //Tensor<data_t> kernel = dynamic_cast<DNNLayerWithKernel<data_t>*>(*it)->get_kernel();
            //Tensor<data_t> new_kernel = kernel;
            //Tensor<data_t> kernel(dynamic_cast<DNNLayerWithKernel<data_t>*>(*it)->get_kernel());

            // apply gradient step

            //new_kernel = new_kernel - grad_update;
            //kernel = kernel - grad_update;
            //std::cout<<"Layer "<<(*it)->get_name()<<" Kernel[0]="<<dynamic_cast<DNNLayerWithKernel<data_t>*>(*it)->get_kernel()[0]<<std::endl;


    }
    
    double get_average_sparsity(){
        return cmprx_accumulated_sparsity / (double)cmprx_number_of_runs;
    }

protected:
    layer_list_t layers;
    Regularizer<data_t>* regularizer;
    float weight_decay;
    int total_weights_size;
    // GASPI
    gaspi::Runtime* gaspi_runtime;
    gaspi::Context* gaspi_context;
    gaspi::segment::Segment* gaspi_segment;
    gaspi::group::Rank gaspi_myRank;
    int gaspi_myRank_int;
    int gaspi_com_tag;
    gaspi::group::Rank gaspi_chiefRank;
    int gaspi_num_ranks;
    // Comprex
    data_t comprex_threshold;
    compressed_exchange::ComprExRunLengths<data_t>* comprex_handle;
    // comprex statistics
    long cmprx_number_of_runs;
    double cmprx_accumulated_sparsity;

    // optimizer dependent update rule
    virtual Tensor<data_t> get_update(Tensor<data_t>*) = 0;

    virtual Tensor<data_t> get_update(DNNLayer<data_t>* base_layer) {
        //DNNLayerWithKernel<data_t>* layer = dynamic_cast<DNNLayerWithKernel<data_t>*>(base_layer);
        Tensor<data_t> grad = dynamic_cast<DNNLayerWithKernel<data_t>*>(base_layer)->get_kernel_grad();
        return get_update(&grad);
    }

};


template<typename data_t>
class SGD_Optimizer : public Comprex_Optimizer<data_t> {
public:
    typedef std::vector<DNNLayer<data_t>*> layer_list_t;

    SGD_Optimizer<data_t>(float learning_rate, gaspi::Runtime& gaspi_runtime, gaspi::Context& gaspi_context)
                : Comprex_Optimizer<data_t>(gaspi_runtime, gaspi_context){
        this->learning_rate = learning_rate;
    }

    void set_learning_rate(float learning_rate){
        this->learning_rate=learning_rate;
    }
protected:
    float learning_rate;
    /*
    virtual Tensor<data_t> get_update(DNNLayer<data_t>* base_layer) {
        DNNLayerWithKernel<data_t>* layer = dynamic_cast<DNNLayerWithKernel<data_t>*>(base_layer);
        return (layer->get_kernel_grad() * learning_rate);
    }*/
    Tensor<data_t> get_update(Tensor<data_t>* grad) {
        return (*grad * learning_rate);
    }
};

#endif //OPTIMIZER_H
