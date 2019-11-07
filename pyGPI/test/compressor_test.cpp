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

    // pseudo communication buffer
    char buffer[10*size];

    // CompressorRLE test
    {
        int errors = 0;
        CompressorRLE<data_t> compressor;
        std::cout<<"Test CompressorRLE class."<<std::endl;

        std::cout<<"Compress..."<<std::endl;
        std::unique_ptr<CompressedVector<data_t> > compressed_vec;
        compressor.compress(&compressed_vec, &values);
        compressed_vec.get()->writeBuffer(reinterpret_cast<void*>(buffer));

        std::cout<<"Decompress..."<<std::endl;
        auto cVect=compressor.getEmptyCompressedVector();
        cVect.get()->loadBuffer(reinterpret_cast<void*>(buffer));
        std::vector<data_t> out_vec;
        compressor.decompress(&out_vec, cVect.get());
        std::cout<<"Done."<<std::endl;


        std::cout<<"Original: #elements: "<<nelem<<" ("<<100.0*(1-(float)nelem/(float)values.size())<<"% sparsity)"<<std::endl;
        std::string compressedVect_info_str = dynamic_cast<const CompressedVectorRLE<data_t>*>(cVect.get())->sprint();
        std::cout<<compressedVect_info_str;
        /*
        if(nelem != compressedVect_size){
            std::cout<<"Number of communicated elements("<<compressedVect_size<<") differs from expected value ("<<nelem<<")!"<<std::endl;
        }*/
        for(int i=0; i<values.size(); ++i){
            if( out_vec[i] != values[i] ) {
                std::cout<<values[i]<<"="<<out_vec[i]<<" failed!"<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    }

    // without getting compressed Vectors
    {
        int errors = 0;
        CompressorRLE<data_t> compressor;
        std::cout<<"Test CompressorRLE class, quick interface without obtaining compressed Vector."<<std::endl;

        std::cout<<"Compress..."<<std::endl;
        compressor.compress(reinterpret_cast<void*>(buffer), &values);

        std::cout<<"Decompress..."<<std::endl;
        std::vector<data_t> out_vec;
        compressor.decompress(&out_vec, reinterpret_cast<void*>(buffer));
        std::cout<<"Done."<<std::endl;
        /*
        if(nelem != compressedVect_size){
            std::cout<<"Number of communicated elements("<<compressedVect_size<<") differs from expected value ("<<nelem<<")!"<<std::endl;
        }*/
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