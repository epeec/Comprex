#include <iostream>
#include <string>
#include <vector>

#include <allreduce.hxx>
#include <allreduce.tpp>
#include <gaspiEnvironment.hxx>
#include <gaspiEnvironment.tpp>


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
    
    const int size = 1000000;
    const int num_runs=10;
    float compression_ratio = 1e-2;
    unsigned long segment_size = static_cast<size_t>(1) << 32;

    // init GaspiCxx Environment
    GaspiEnvironment gaspi_env(segment_size); 

    int myRank = gaspi_env.get_rank();
    int numRanks = gaspi_env.get_ranks();

    int chiefRank = 0;
    int tag = 0;

    // generate test data
    std::vector<data_t> values(size);
    for(int i=0; i<values.size(); ++i) {
        values[i] = i * (1-2*(i%2));
    }

    // generate gold results
    std::vector<data_t> gold_result(size);
    for(int i=0; i<gold_result.size(); ++i) {
        gold_result[i] = (num_runs*(numRanks-1)+1)*values[i];
    }

    // results vector
    std::vector<data_t> result(size);
    for(int i=0; i<result.size(); ++i) {
        result[i] = values[i];
    }

    // setup Allreduce
    std::cout<<"Building Allreduce"<<std::endl;
    Comprex_AllToOneAllreduce<data_t> allreduce(&gaspi_env);

    // define connectivity pattern
    std::cout<<"Connecting Nodes"<<std::endl;
    float size_factor = 3.0;
    allreduce.setupConnections(size, tag, chiefRank, compression_ratio, size_factor);

    // MAIN LOOP
    std::cout<<"Executing Main"<<std::endl;
    int progress_run=0;
    for(int run = 0; run<num_runs; run++) { 

        if(myRank == chiefRank and num_runs >=10) {
            if((run-progress_run)%(num_runs/10)==0){
                std::cout<<((float)run/num_runs)*100<<"\%..."<<std::endl;
                progress_run = run;
            }
        }
        if(myRank != chiefRank){
            for(int i=0; i<size; ++i){
                result[i] = values[i];
            }
        }
        allreduce.apply(result.data(), size);
    } //MAIN LOOP
    allreduce.flush(result.data(), size);

    // check for errors
    if(myRank == chiefRank){
        int errors=0;
        for(int i=0; i<result.size(); ++i){
            if(result[i] != gold_result[i]) {
                std::cout<<"Rank "<<myRank<<" Error in ["<<i<<"]: result: "<<result[i]<<" gold: "<<gold_result[i]<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    } 
    gaspi_env.barrier();
    return 0;
}