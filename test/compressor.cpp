#include <compressor.hxx>

#include <iostream>
#include <string>
#include <vector>


typedef int data_t;

void print_pass(int errors){
    if(errors == 0) std::cout<<"+++ PASSED +++"<<std::endl;
    else {
        std::cout<<"--- FAILED ---"<<std::endl;
        std::cout<<"with "<<errors<<" errors."<<std::endl;
    }
}

template<typename data_t>
void print_vector(std::string name, const std::vector<data_t>& vect){
    std::cout<<name<<": [";
    for(int i=0; i<vect.size(); ++i){
        std::cout<<vect[i]<<", ";
    }
    std::cout<<"\b\b]"<<std::endl;
}

int main() {
    // generate test data
    const int size = 100;
    std::vector<data_t> values(size);
    for(int i=0; i<values.size(); ++i) {
        values[i] = i * ((i%4)/3)*(1-2*((i%6)/5)); // generate 0 elements
    }
    int nelem=0;
        for(auto e : values){
        if(e==0) ++nelem;
    }
    nelem = values.size()-nelem;

    print_vector("Input data:", values);

    // pseudo communication buffer
    char buffer[10*size];

    // CompressorRLE test
    {
        int errors = 0;
        std::vector<data_t> myValues(values);
        CompressorRLE<data_t> compressor;
        std::cout<<"Test CompressorRLE class."<<std::endl;
        std::cout<<"Compress..."<<std::endl;
        compressor.compress(reinterpret_cast<void*>(buffer), myValues.data(), myValues.size(), 0);

        std::cout<<"Decompress..."<<std::endl;
        std::vector<data_t> out_vec(size);
        compressor.decompress(out_vec.data(), out_vec.size(), reinterpret_cast<void*>(buffer));
        std::cout<<"Done."<<std::endl;

        std::cout<<"Original: #elements: "<<nelem<<" ("<<100.0*(1-(float)nelem/(float)values.size())<<"% sparsity)"<<std::endl;

        for(int i=0; i<values.size(); ++i){
            if( out_vec[i] != values[i] ) {
                std::cout<<values[i]<<"="<<out_vec[i]<<" failed!"<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    }

    // CompressorIndexPairs test
    {
        int errors = 0;
        std::vector<data_t> myValues(values);
        CompressorIndexPairs<data_t> compressor(0.01);
        std::cout<<"Test CompressorIndexPairs class."<<std::endl;

        std::cout<<"Compress..."<<std::endl;
        compressor.compress(reinterpret_cast<void*>(buffer), myValues.data(), myValues.size(), 0);

        std::cout<<"Decompress..."<<std::endl;
        std::vector<data_t> out_vec(size);
        compressor.decompress(out_vec.data(), out_vec.size(), reinterpret_cast<void*>(buffer));
        std::cout<<"Done."<<std::endl;

        std::cout<<"Original: #elements: "<<nelem<<" ("<<100.0*(1-(float)nelem/(float)values.size())<<"% sparsity)"<<std::endl;

        for(int i=0; i<values.size(); ++i){
            if( out_vec[i] != values[i] ) {
                std::cout<<values[i]<<"="<<out_vec[i]<<" failed!"<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    }


    return 0;
}