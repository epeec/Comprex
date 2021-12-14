#include <gaspiExchange.hxx>

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
    // generate test data
    int size = 100;
    
    // init GaspiCxx
    gaspi::Runtime runtime;
    gaspi::Context context;
    gaspi::segment::Segment segment( static_cast<size_t>(1) << 28);
    gaspi::group::Rank myRank =  context.rank();
    gaspi::group::Rank srcRank(0);
    gaspi::group::Rank destRank(1);
    int tag = 1;  // comm tag

    // setup GaspiEx
    std::cout<<"Building GaspiEx"<<std::endl;
    GaspiEx<int> gaspiEx(runtime, context, segment, size);

    if(myRank == srcRank){
        gaspiEx.connectTx(destRank, tag);
    }
    else {
        gaspiEx.connectRx(srcRank, tag);
    }

    if(myRank == srcRank) {
        std::cout<<"Rank "<<srcRank.get()<<" sends data to Rank "<<destRank.get()<<"."<<std::endl;
    }

    const int num_runs=3;
    for(int run = 0; run<num_runs; run++) { 
        context.barrier();

        std::vector<data_t> values(size);
        for(int i=0; i<values.size(); ++i) {
            values[i] = i * (1-2*(i%2));
        }

        // Source side
        // ------------------------------------------------------------
        if(myRank == srcRank) {
            std::cout<<"Run #"<<run<<std::endl;
            std::cout<<"========================================="<<std::endl;
        
            // send data to Receiver side.
            // write test data
            print_vector<data_t>("Source Vector", values);
            gaspiEx.writeRemote(values, destRank, tag);
        } 

        // Destination side
        // ------------------------------------------------------------
        if(myRank == destRank) {
            // get Data from sender
            std::vector<data_t> rxVect(2*size);
            int recv_size;
            recv_size = gaspiEx.readRemote(rxVect, srcRank, tag);
            rxVect.resize(recv_size);
            print_vector<data_t>("Received Vector", rxVect);
        }
        context.barrier();
        size /=2;
    } //run
    return 0;
}