#include <iostream>
#include <string>
#include <vector>

#include <comprEx.hxx>
#include <comprEx.tpp>

#include <gaspiEnvironment.hxx>
#include <gaspiEnvironment.tpp>

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
    
    const int size = 500000;
    const int num_runs=10000;
    size_t segment_size = static_cast<size_t>(1) << 28;

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

    // create Timer
    auto timer = TictocTimer();

    // init GaspiCxx
    GaspiEnvironment gaspi_env(segment_size);
    int myRank =  gaspi_env.get_rank();
    int srcRank = 0;
    int destRank = 1;
    int tag = 1;  // comm tag

    // setup comprex
    std::cout<<"Building ComprEx"<<std::endl;
    // ComprEx must know the size of the transmission in order to handle Rests
    ComprEx<int> cmprex(gaspi_env.get_segment(), size, 3);
    cmprex.setThreshold(ThresholdTopK<data_t>(0.01));
    // cmprex.setCompressor(CompressorRLE<data_t>());
    // cmprex.setThreshold(ThresholdNone<data_t>());
    cmprex.setCompressor(CompressorIndexPairs<data_t>());
    // cmprex.setCompressor(CompressorNone<data_t>());

    // define connectivity pattern
    cmprex.connectTo( (myRank+1)%2, (myRank+1)%2, tag);
    // if(myRank == srcRank){
    //     cmprex.connectTx(destRank, size, tag);
    // } else {
    //     cmprex.connectRx(srcRank, size, tag);
    // }

    if(myRank == srcRank) {
        std::cout<<"Rank "<<srcRank<<" sends data to Rank "<<destRank<<"."<<std::endl;
    }

    int progress_run=0;

    timer.tic();
    for(int run = 0; run<num_runs; run++) { 
        // Source side
        // ------------------------------------------------------------
        if(myRank == srcRank) {
            // std::cout<<"Run #"<<run<<std::endl;
            // std::cout<<"========================================="<<std::endl;

            if((run-progress_run)%(num_runs/10)==0){
                std::cout<<((float)run/num_runs)*100<<"\%..."<<std::endl;
                progress_run = run;
            }
            
            // std::vector<data_t> rests = cmprex.getRests();
            // send data to Receiver side. In last iteration send remaining rests.
            if(run<num_runs-1){
                // write test data
                // print_vector<data_t>("Source Vector", values);
                // print_vector<data_t>("Rests Vector:", rests);
                cmprex.writeRemote(values.data(), values.size());
            }
            else {
                // at the end, flush out the rests
                // print_vector<data_t>("Flush Rests Vector:", rests);
                cmprex.flushRests();
            }
            // print Rests Vector after send, because it should be updated
            // rests = cmprex.getRests();
            // print_vector<data_t>("Rests Vector after send:", rests);
        } 

        // Destination side
        // ------------------------------------------------------------
        if(myRank == destRank) {
            // get Data from sender
            std::vector<data_t> rxVect(size);
            cmprex.readRemote(rxVect.data(), rxVect.size());
            //print_vector<data_t>("Received Vector:", rxVect);
            for(int i=0; i<rxVect.size(); ++i){
                result[i] += rxVect[i]; 
            }
        }// if(myRank == destRank)
        gaspi_env.barrier();
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