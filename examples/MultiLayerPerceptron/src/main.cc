#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <math.h>
#include "DNN.pb.h"
#include "read_MNIST.h"
#include "layers.h"
#include "MLP.h"

static const int epochs=20;
static const std::string ckpt_basename = "mlp_train";
static const float init_learning_rate=0.0001;
static const int batch_size=128;

void print_proto(std::string pb_name) {
    myDNN::Net data_proto;
    std::fstream input(pb_name, std::ios::in | std::ios::binary);
    if (!data_proto.ParseFromIstream(&input)) {
        throw std::runtime_error("Failed to parse protobuf "+pb_name);
    }

    std::cout<<data_proto.name()<<std::endl;
    std::cout<<"############################"<<std::endl;
    for(int layer=0; layer < data_proto.layers_size(); ++layer) {
        std::cout<<std::endl;
        std::cout<<"layer: "<<data_proto.layers(layer).name()<<std::endl;
        std::cout<<"shape: [";
        for(int dim=0; dim<data_proto.layers(layer).shape_size(); ++dim){
            std::cout<<data_proto.layers(layer).shape(dim)<<", ";
        } std::cout<<"]"<<std::endl;
        std::cout<<std::endl;
    }
}

int main(int argc, char *argv[]) {

    // initialize GASPI
    gaspi::Runtime gaspi_runtime;
    gaspi::Context gaspi_context;
    gaspi::group::Rank gaspi_myRank = gaspi_context.rank();
    int gaspi_num_ranks = gaspi_context.size().get();
    gaspi::group::Rank gaspi_chiefRank(0);

    // load MNIST
    if(argc == 1) {
        std::cerr<<"Missing argument for MNIST path!"<<std::endl;
        return -1;
    }
    if(argc == 2) {
        std::cerr<<"Missing argument for checkpoint path!"<<std::endl;
        return -1;
    }

    std::string mnist_path = argv[1];
    MNISTData mnist(mnist_path+"/train-images-idx3-ubyte",mnist_path+"/train-labels-idx1-ubyte");

    std::string ckpt_dir = argv[2];

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // input and data buffers
    typedef float tensor_t;

    int tensor_shape[4]={batch_size, mnist.get_height(), mnist.get_width(), 1};
    Tensor<tensor_t> images(tensor_shape, 4);

    int n_classes = mnist.get_num_classes();
    int label_shape[4]={batch_size, n_classes, 1, 1};
    Tensor<tensor_t> labels(label_shape, 4); // 1-hot labels
    int disp_labels[batch_size]; // labels for printing

    // Build MLP
    //print_proto("MLP.pb");
    if(gaspi_myRank==gaspi_chiefRank) std::cout<<"Building MLP..."<<std::endl;
    MLP<float> network(images.get_shape());
    //lenet.load_checkpoint("LeNet2.pb");
    //lenet.save_checkpoint("LeNet2.pb");

    //mlp.set_optimizer(0.1/(float)gaspi_num_ranks/(float)batch_size);
    Comprex_Optimizer<tensor_t>* optimizer = new SGD_Optimizer<tensor_t>(init_learning_rate, gaspi_runtime, gaspi_context);
    network.set_optimizer(optimizer);

    //lenet.print_proto();
    //lenet.print_net();

    // Output tensor
    Tensor<tensor_t> output = Tensor<tensor_t>();

    int num_steps = epochs*mnist.get_num_images()/batch_size/gaspi_num_ranks;
    int dataset_offset= gaspi_myRank.get()*mnist.get_num_images()/gaspi_num_ranks;
    for(int batch_it=1; batch_it<=num_steps; ++batch_it) {
        // load input and label
        // transform labels to 1-hot
        for(int b=0; b<batch_size; ++b) {
            int dataset_idx=(dataset_offset+b)%mnist.get_num_images();
            disp_labels[b] = mnist.get_entry(dataset_idx)->label;
            // 1-hot conversion
            for(int i=0; i<n_classes; ++i) {
                int label_idx = b*n_classes + i;
                labels[label_idx]= (mnist.get_entry(dataset_idx)->label == i) ? 1.0 : 0.0;
            }

            // input images
            for(int i=0; i<mnist.get_entry(dataset_idx)->size;++i) {
                int tensor_idx = i + b * tensor_shape[1]*tensor_shape[2]*tensor_shape[3];
                images[tensor_idx] = (tensor_t)(mnist.get_entry(dataset_idx)->image[i])/128.0-1.0;
            }
        }
        dataset_offset = (dataset_offset+batch_size)%mnist.get_num_images();

        /*
        int e=0;
        for(int y=0; y<28; ++y) {
            for(int x=0; x<28; ++x) {
                if(images[y*28+x+784*e]<0)
                    std::cout<<".";
                else
                    std::cout<<"#";
            }
            std::cout<<endl;
        }
        std::cout<<"label: "<<disp_labels[e]<<std::endl;
        */

        if(gaspi_myRank==gaspi_chiefRank) std::cout<<"Batch #"<<batch_it<<" of "<<num_steps<<std::endl
                 <<"--------------------------"<<std::endl;
        if(gaspi_myRank==gaspi_chiefRank) std::cout<<"Inferencing..."<<std::endl;
        output = network.forward(images, labels);
        if(gaspi_myRank==gaspi_chiefRank) std::cout<<"Backpropagating..."<<std::endl;
        network.backprop();
         if(gaspi_myRank==gaspi_chiefRank) std::cout<<"Updating..."<<std::endl;
        network.update();

        // learning rate decay
        if((batch_it)%100==0){
            network.decay_learning_rate(1.0);
        }

        // checkpointing
        if(gaspi_myRank==gaspi_chiefRank){
            if(batch_it%100==1){
                std::string ckpt_name = ckpt_basename+"_"+std::to_string(batch_it)+".pb";
                std::cout<<"Saving checkpoint "<<ckpt_name<<"..."<<std::endl;
                network.save_checkpoint(ckpt_dir+ckpt_name);
            }
        }
        // output results
        float accuracy = 0.0;
        for(int b=0; b<batch_size; ++b){
            //std::cout<<"Batch "<<b<<" - Estimate: "<<output[b]<<" GT: "<<disp_labels[b]<<std::endl;
            if(output[b] == disp_labels[b]) accuracy += 1.0;
        }
        accuracy /= batch_size;

        //if(gaspi_myRank==gaspi_chiefRank) std::cout<<"Accuracy: "<<accuracy<<std::endl;
        //if(gaspi_myRank==gaspi_chiefRank) std::cout<<std::endl;
        std::cout<<"Rank "<<gaspi_myRank.get()<<" Accuracy: "<<accuracy<<std::endl;
        std::cout<<std::endl;
        //if(accuracy>=0.9) break;
    }//batch_it

    std::cout<<"Rank "<<gaspi_myRank.get()<<" Average Sparsity: "<<optimizer->get_average_sparsity()<<std::endl;


    // Final test Accuracy
    if(gaspi_myRank==gaspi_chiefRank) {
        MNISTData mnist(mnist_path+"/t10k-images-idx3-ubyte",mnist_path+"/t10k-labels-idx1-ubyte");
        int num_steps = mnist.get_num_images()/batch_size;
        int dataset_offset=0;
        float test_accuracy = 0.0;

        for(int batch_it=1; batch_it<=num_steps; ++batch_it) {
            // load input and label
            // transform labels to 1-hot
            for(int b=0; b<batch_size; ++b) {
                int dataset_idx=(dataset_offset+b)%mnist.get_num_images();
                disp_labels[b] = mnist.get_entry(dataset_idx)->label;
                // 1-hot conversion
                for(int i=0; i<n_classes; ++i) {
                    int label_idx = b*n_classes + i;
                    labels[label_idx]= (mnist.get_entry(dataset_idx)->label == i) ? 1.0 : 0.0;
                }

                // input images
                for(int i=0; i<mnist.get_entry(dataset_idx)->size;++i) {
                    int tensor_idx = i + b * tensor_shape[1]*tensor_shape[2]*tensor_shape[3];
                    images[tensor_idx] = (tensor_t)(mnist.get_entry(dataset_idx)->image[i])/128.0-1.0;
                }
            }
            dataset_offset = (dataset_offset+batch_size)%mnist.get_num_images();

            output = network.forward(images, labels);

            for(int b=0; b<batch_size; ++b){
                //std::cout<<"Batch "<<b<<" - Estimate: "<<output[b]<<" GT: "<<disp_labels[b]<<std::endl;
                if(output[b] == disp_labels[b]) test_accuracy += 1.0;
            }
        }
        test_accuracy /= (batch_size*num_steps);
        std::cout<<"Test Accuracy: "<<test_accuracy<<std::endl;
    }
    
    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
