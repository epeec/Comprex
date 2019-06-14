#ifndef MLP_H
#define MLP_H

#include "layers.h"
#include "optimizer.h"
#include "DNN.pb.h"
#include <vector>
#include "comprex.hxx"

template<typename tensor_t>
class MLP {
public:
    MLP<tensor_t>(std::vector<int> input_shape){
        build_net(input_shape);
        this->optimizer=NULL;
        optimizer_lr=0;
    }

    ~MLP<tensor_t>(){
        for(typename net_t::iterator it = net.begin(); it != net.end(); ++it) {
            delete *it;
        }
        delete prediction;
        delete loss;
        if(optimizer != NULL){
            delete optimizer;
        }
    }

    void load_checkpoint(const std::string& pb_name){
        myDNN::Net data_proto; // protobuf containing pre-trained model
        open_protobuf(data_proto, pb_name);
        for(typename net_t::iterator it = net.begin(); it != net.end(); ++it) {
            if((*it)->has_kernel()){
                load_layer(dynamic_cast<DNNLayerWithKernel<tensor_t>*>(*it), data_proto);
            }
        }
    }

    void save_checkpoint(const std::string& pb_name){
        myDNN::Net data_proto; // protobuf which will be filled with current model
        data_proto.set_name("MLP");
        for(typename net_t::iterator it = net.begin(); it != net.end(); ++it) {
            if((*it)->has_kernel()){
                DNNLayerWithKernel<tensor_t>* data_layer = dynamic_cast<DNNLayerWithKernel<tensor_t>*>(*it);
                Tensor<tensor_t> kernel = data_layer->get_kernel();
                myDNN::Net_Layer* proto_layer = data_proto.add_layers();
                proto_layer->set_name(data_layer->get_name());
                for(int i=0; i< kernel.get_dims(); ++i){
                    proto_layer->add_shape(kernel.get_shape(i));
                }
                for(int i=0; i< kernel.get_size(); ++i){
                    proto_layer->add_data(kernel.get_data(i));
                }
                // Write DNN
                 fstream output(pb_name, ios::out | ios::trunc | ios::binary);
                 if (!data_proto.SerializeToOstream(&output)) {
                    throw std::runtime_error("Failed to write protobuf "+pb_name);
                 }

            }
        }

    }

    void set_optimizer(Comprex_Optimizer<tensor_t>* new_optimizer){
        if(optimizer != NULL){
            delete optimizer;
        }
        this->optimizer = new_optimizer;
        this->optimizer->set_layers(this->net);
        this->optimizer->set_regularizer("L2", 0.00005);
    }

    void decay_learning_rate(float factor){
        optimizer_lr *= factor;
        dynamic_cast<SGD_Optimizer<tensor_t>*>(optimizer)->set_learning_rate(optimizer_lr);
    }

    Tensor<tensor_t> forward(Tensor<tensor_t> tensor, Tensor<tensor_t> labels) {
        for(typename net_t::iterator it = net.begin(); it != net.end(); ++it) {
            tensor = (*it)->forward(tensor);
        }
        float loss = this->loss->forward(tensor, labels);
        std::cout<<"loss: "<<loss<<std::endl;
        tensor =  this->prediction->forward(tensor);
        return tensor;
    }

    void backprop() {
        Tensor<tensor_t> tensor = this->loss->get_backprop_grad();
        //std::cout<<"Loss: "<<tensor.shape_string()<<std::endl;
        for(typename net_t::reverse_iterator it = net.rbegin(); it != net.rend(); ++it) {
            tensor = (*it)->backprop(tensor);
            //std::cout<<"Layer "<<(*it)->get_name()<<" Gradient[0]="<<tensor.get_data(0)<<std::endl;
            //std::cout<<(*it)->get_name()<<": "<<tensor.shape_string()<<std::endl;
        }
    }

    void update() {
        if(optimizer == NULL){
            throw std::runtime_error("Optimizer has not been initialized!");
        }
        this->optimizer->apply();
    }

    //TODO: training loop, switch dropout layer
    void train_for(int steps){

    }


    void print_net() {
        for(typename net_t::iterator it = net.begin(); it != net.end(); ++it) {
            std::cout<<(*it)->get_name()<<": "<<(*it)->output_shape_string()<<std::endl;
        }
    }

protected:
    typedef std::vector<int> shape_t;
    typedef std::vector<DNNLayer<tensor_t>*> net_t;

    net_t net; // a ordered list of layers
    DNNLayer<tensor_t>* prediction;
    LossLayer<tensor_t>* loss;
    Comprex_Optimizer<tensor_t>* optimizer;

    float optimizer_lr;

    void load_layer(DNNLayerWithKernel<tensor_t>* target, myDNN::Net& data_proto) {
        int layer=0;
        // find layer
        for(int i=0; i<data_proto.layers_size(); ++i) {
            if(data_proto.layers(i).name().find(target->get_name()) != std::string::npos ) break;
            ++layer;
        }
        if(layer==data_proto.layers_size()){
            const std::string message="Could not initialize layer " + target->get_name() + " from checkpoint, layer not found!";
            throw std::runtime_error(message);
        }

        // check shapes
        if(data_proto.layers(layer).shape_size() > target->get_kernel().get_dims()){
            const std::string message="Could not initialize layer " + target->get_name() + " from checkpoint, layers have different dimensions!";
            throw std::runtime_error(message);

        }
        for(int i=0; i<data_proto.layers(layer).shape_size(); ++i){
            if(data_proto.layers(layer).shape(i) != target->get_kernel().get_shape(i)){
                const std::string message="Could not initialize layer " + target->get_name() + " from checkpoint, layers have different shapes!";
                throw std::runtime_error(message);
            }

        }

        Tensor<tensor_t> kernel( data_proto.layers(layer).data(), data_proto.layers(layer).shape() );
        kernel.reshape(target->get_kernel().get_shape());
        target->set_kernel(kernel);
    }

    void build_net(shape_t input_shape){

        // MLP
        // flatten
        net.push_back( new Flatten<tensor_t>(input_shape, "MLP/Flatten") );

        // fc 1
        input_shape = net.back()->get_output_shape();
        fully_connected_layer(input_shape, {128}, "MLP/fc1");

        // dropout
        //input_shape = net.back()->get_output_shape();
        //net.push_back( new Dropout<tensor_t>(input_shape, 0.5, "MLP/Dropout") );
        //dynamic_cast<Dropout<tensor_t>*>(net.back())->enable_training();

        // fc 2
        input_shape = net.back()->get_output_shape();
        fully_connected_layer(input_shape, {64}, "MLP/fc2");

        // fc 3
        input_shape = net.back()->get_output_shape();
        fully_connected_layer(input_shape, {10}, "MLP/fc3");

        // Outputs
        input_shape = net.back()->get_output_shape();
        // Argmax (Prediction)
        this->prediction = new Argmax<tensor_t>(input_shape, "MLP/Logits");
        // Softmax-Crossentropy (Loss)
        this->loss = new SMCELoss<tensor_t>(input_shape, "MLP/Loss");
    }

    void open_protobuf(myDNN::Net& data_proto, std::string pb_name) {
        // Read net
        std::fstream input(pb_name, std::ios::in | std::ios::binary);
        if (!data_proto.ParseFromIstream(&input)) {
            throw std::runtime_error("Failed to parse protobuf "+pb_name);
        }
    }

    //kernel_shape=[height, width, out_channels]
    void conv2d_layer(shape_t in_shape, shape_t kernel_shape, string layer_name){
        //HeInitializer<tensor_t> init_weight;
        NormalTruncInitializer<tensor_t> init_weight(0.1);
        ZeroInitializer<tensor_t> init_bias;
        int stride = 1;
        Padding padding = Padding::same;

        // conv2d
        net.push_back( new Conv2D<tensor_t>(in_shape, kernel_shape, &init_weight, stride, padding, layer_name+"/weights") );

        // add bias + ReLU
        // output does not change shape
        // element-wise operations, so input and output tensor can be equal.
        in_shape = net.back()->get_output_shape();
        net.push_back( new AddBias<tensor_t>(in_shape, &init_bias, layer_name+"/biases") );
        in_shape = net.back()->get_output_shape();
        net.push_back( new ReLU<tensor_t>(in_shape, layer_name+"/relu") );
    }

    //kernel_shape=[out_channels]
    void fully_connected_layer(shape_t in_shape, shape_t kernel_shape, string layer_name){
        //HeInitializer<tensor_t> init_weight;
        NormalTruncInitializer<tensor_t> init_weight(0.1);
        ZeroInitializer<tensor_t> init_bias;

        // fc
        net.push_back( new FullyConnected<tensor_t>(in_shape, kernel_shape, &init_weight, layer_name+"/weights") );

        // add bias + ReLU
        in_shape = net.back()->get_output_shape();
        net.push_back( new AddBias<tensor_t>(in_shape, &init_bias, layer_name+"/biases") );
        in_shape = net.back()->get_output_shape();
        net.push_back( new ReLU<tensor_t>(in_shape, layer_name+"/relu") );
    }
};


#endif //MLP_H
