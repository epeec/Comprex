#include <comprEx.hxx>
#include <comprEx.tpp>

#include <iostream>
#include <string>
#include <vector>

#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/segment/Allocator.hpp>
#include <GaspiCxx/segment/NotificationManager.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
#include <GaspiCxx/utility/ScopedAllocation.hpp>

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

int main(){
    
    const int size = 50;
    const int num_runs=4;
    const int size_factor = 3;
    const float threshold = 0.1;

    // generate test data
    std::vector<data_t> values(size);
    for(int i=0; i<values.size(); ++i) {
        values[i] = i * (1-2*(i%2));
    }

    // generate gold results
    // there are num_runs-1 effective runs!
    std::vector<data_t> gold_result(size);
    for(int i=0; i<gold_result.size(); ++i) {
        gold_result[i] = (num_runs-1)*values[i];
    }

    // results vector
    std::vector<data_t> result(size);
    for(int i=0; i<result.size(); ++i) {
        result[i] = 0;
    }

    // init GaspiCxx
    gaspi::initGaspiCxx();
    gaspi::segment::Segment* segment = new gaspi::segment::Segment( static_cast<size_t>(1) << 28);
    gaspi::group::Rank myRank =  gaspi::group::Group().rank();
    gaspi::group::Rank srcRank(0);
    gaspi::group::Rank destRank(1);
    int tag = 1;  // comm tag

    // setup comprex
    std::cout<<"Rank "<<myRank.get()<<" building ComprEx"<<std::endl;
    // ComprEx must know the size of the transmission in order to handle Rests
    ComprEx<int> cmprex(segment, size, size_factor, true);
    cmprex.setThreshold(ThresholdTopK<data_t>(threshold));
    cmprex.setCompressor(CompressorIndexPairs<data_t>());

    // define connectivity pattern
    cmprex.connectTo( gaspi::group::Rank((myRank.get()+1)%2), gaspi::group::Rank((myRank.get()+1)%2), tag);

    if(myRank == srcRank) {
        std::cout<<"Rank "<<srcRank.get()<<" sends data to Rank "<<destRank.get()<<"."<<std::endl;
    }

    int progress_run=0;

    for(int run = 0; run<num_runs; run++) { 
        // Source side
        // ------------------------------------------------------------
        if(myRank == srcRank) {
            // send data to Receiver side. In last iteration send remaining rests.
            if(run<num_runs-1){
                // write test data
                cmprex.writeRemote(values.data(), values.size());
            }
            else {
                // at the end, flush out the rests
                cmprex.flushRests();
            }
        } 

        // Destination side
        // ------------------------------------------------------------
        if(myRank == destRank) {
            // get Data from sender
            cmprex.readRemote_add(result.data(), result.size());
        }

        gaspi::getRuntime().barrier();
    } //run

    if(myRank == destRank) {
        int errors=0;
        for(int i=0; i<result.size(); ++i){
            if(result[i] != gold_result[i]) {
                std::cout<<"Error in ["<<i<<"]: result: "<<result[i]<<" gold: "<<gold_result[i]<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    }
    return 0;
}