#include <gaspiEx.hxx>
#include <gaspiEx.tpp>

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

#include <tictocTimer.hxx>


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
    // init gaspiCxx
    gaspi::initGaspiCxx();
    
    const int size = 500000;
    const int num_runs=10000;

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

    // create Timer
    auto timer = TictocTimer();

    // init GaspiCxx
    gaspi::segment::Segment* segment = new gaspi::segment::Segment( static_cast<size_t>(1) << 30);
    gaspi::group::Rank myRank = gaspi::group::Group().rank();
    gaspi::group::Rank srcRank(0);
    gaspi::group::Rank destRank(1);
    int tag = 1;  // comm tag

    // setup gaspiEx
    GaspiEx<int> gaspiex(segment, size);

    // define connectivity pattern
    gaspiex.connectTo( gaspi::group::Rank((myRank.get()+1)%2), gaspi::group::Rank((myRank.get()+1)%2), tag);
    // if(myRank == srcRank){
    //     cmprex.connectTx(destRank, tag);
    // } else {
    //     cmprex.connectRx(srcRank, tag);
    // }

    if(myRank == srcRank) {
        std::cout<<"Rank "<<srcRank.get()<<" sends data to Rank "<<destRank.get()<<"."<<std::endl;
    }

    int progress_run=0;

    timer.tic();
    for(int run = 0; run<num_runs; run++) { 
        // Source side
        // ------------------------------------------------------------
        if(myRank == srcRank) {
            // std::cout<<"Run #"<<run<<std::endl;
            // std::cout<<"========================================="<<std::endl;
            
            // std::vector<data_t> rests = cmprex.getRests();
            // send data to Receiver side. In last iteration send remaining rests.

            if((run-progress_run)%(num_runs/10)==0){
                std::cout<<((float)run/num_runs)*100<<"\%..."<<std::endl;
                progress_run = run;
            }

            if(run<num_runs){
                // write test data
                // print_vector<data_t>("Source Vector", values);
                // print_vector<data_t>("Rests Vector:", rests);
                gaspiex.writeRemote(values.data(), values.size());
            }
            else {
                // do nothing
            }
        } 

        // Destination side
        // ------------------------------------------------------------
        if(myRank == destRank && run<num_runs) {
            // get Data from sender
            std::vector<data_t> rxVect(size);
            gaspiex.readRemote(rxVect.data(), rxVect.size());
            //print_vector<data_t>("Received Vector:", rxVect);
            for(int i=0; i<rxVect.size(); ++i){
                result[i] += rxVect[i]; 
            }
        }// if(myRank == destRank)
        gaspi::getRuntime().barrier();
    } //run
    timer.toc();

    gaspi_printf("Runtime: %.2f ms\n",timer.get_time_us()*1e-3);

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