#include <compressedExchange.hxx>

#include <iostream>
#include <string>
#include <vector>

#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/Context.hpp>
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
    /////////////////////////////////////
    // Parameters
    const int size = 500000; // size of vector
    const int num_runs=10000; // number of communication rounds

    const float topK = 0.01; // compression topK value
    /////////////////////////////////////

    // generate test data
    std::vector<data_t> values(size);
    for(int i=0; i<values.size(); ++i) {
        values[i] = i * (1-2*(i%2));
    }

    // generate gold results
    std::vector<data_t> gold_result(size);
    for(int i=0; i<gold_result.size(); ++i) {
        gold_result[i] = (num_runs)*values[i];
    }

    // results vector
    std::vector<data_t> result(size);
    for(int i=0; i<result.size(); ++i) {
        result[i] = 0;
    }

    // init GaspiCxx
    gaspi::Runtime* runtime = new gaspi::Runtime();
    gaspi::Context* context = new gaspi::Context();
    gaspi::segment::Segment* segment = new gaspi::segment::Segment( static_cast<size_t>(1) << 28);
    gaspi::group::Rank myRank =  context->rank();
    gaspi::group::Rank srcRank(0); // Rank which sends the data
    gaspi::group::Rank destRank(1); // Rank which receives the data
    int tag = 1;  // communication tag

    // setup comprex
    // ComprEx must know the size of the transmission in order to handle Rests
    ComprEx<int> cmprex(*runtime, *context, *segment, size);
    cmprex.setThreshold(ThresholdTopK<data_t>(topK));
    cmprex.setCompressor(CompressorRLE<data_t>());

    // define connectivity pattern
    cmprex.connectTo( gaspi::group::Rank((myRank.get()+1)%2), gaspi::group::Rank((myRank.get()+1)%2), tag, 2);

    if(myRank == srcRank) {
        std::cout<<"Rank "<<srcRank.get()<<" sends data to Rank "<<destRank.get()<<"."<<std::endl;
    }

    int progress_run=0; // for progress messages

    ////////////////////////////////
    // main communication loop
    ////////////////////////////////
    for(int run = 0; run<num_runs+1; run++) { // num_runs + 1 for flushing in the last step
        // Source side
        // ------------------------------------------------------------
        if(myRank == srcRank) {
            // display progress
            if((run-progress_run)%(num_runs/10)==0){
                std::cout<<((float)run/num_runs)*100<<"\%..."<<std::endl;
                progress_run = run;
            }
            
            if(run<num_runs){
                cmprex.writeRemote(&values);
            }
            else {
                cmprex.flushRests();
            }
        } 

        // Destination side
        // ------------------------------------------------------------
        if(myRank == destRank) {
            // get Data from sender
            std::vector<data_t> rxVect(size);
            cmprex.readRemote(&rxVect);
            // add data to the results vector
            for(int i=0; i<rxVect.size(); ++i){
                result[i] += rxVect[i]; 
            }
        }
        context->barrier();
    } //run

    // check communicated values against gold result
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