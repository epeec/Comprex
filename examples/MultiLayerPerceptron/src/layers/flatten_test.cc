#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <math.h>
#include <exception>
#include <eigen3/Eigen/Dense>
#include "Layer.pb.h"
#include "layers.h"
#include "tensor.h"
#include "utils/error_checking.h"
#include <limits>
#include "test_utils.h"

static const int BUFFER_SIZE=100000;

const std::string test_data_folder="test_data/";

int main() {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::cout<<"Flatten"<<std::endl;
    std::cout<<"############################"<<std::endl;

    bool is_pass=true;
    int error_count=0;

    int shape[4] = {32,64,64,128};
    int gold_shape[4] = {shape[0],shape[1]*shape[2]*shape[3],1,1};
    flatten(shape, shape);
    
    compare_buffers<int>(is_pass, shape, gold_shape, 4, "Flatten");

    print_pass(is_pass);
 
    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
