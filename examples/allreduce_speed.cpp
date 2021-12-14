#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <experimental/random>
#include <random>
#include <cmath>

#include <allreduce.hxx>
#include <allreduce.tpp>
#include <gaspiEnvironment.hxx>
#include <gaspiEnvironment.tpp>

#include "tictocTimer.hxx"

namespace AllreduceType{
    enum AllreduceType { AllToOne=0, Comprex_AllToOne, Ring, Comprex_Ring, BigRing, Comprex_BigRing };
}

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

template<typename data_t>
float mean(const std::vector<data_t>& values){
    data_t accu=0;
    for(int it=0; it<values.size(); ++it){
        accu += values[it];
    }
    float mean = (float)accu / (float)(values.size());
    return mean;
}

template<typename data_t>
float stddev(const std::vector<data_t>& values){
    float accu=0;
    float mmean=mean(values);
    for(int it=0; it<values.size(); ++it){
        accu += ((float)values[it] - mmean) * ((float)values[it] - mmean);
    }
    float _stddev = std::sqrt(accu / (float)(values.size()-1));
    return _stddev;
}

template<typename data_t>
float stddev_mean(const std::vector<data_t>& values){
    float _stddev = stddev<data_t>(values);
    float stddev_mean = _stddev / std::sqrt((float)(values.size()));
    return stddev_mean;
}

template<typename data_t>
float sparsity(const std::vector<data_t>& values){
    int count=0;
    for(int it=0; it<values.size(); ++it){
        if(values[it]==0) ++count;
    }
    float result = (float)count / (float)values.size();
    return result;
}


int main(int argc, char* argw[]){
    // select allreduce type, if specified
    AllreduceType::AllreduceType allreduce_type = AllreduceType::Comprex_Ring;
    if(argc > 1){
        allreduce_type = (AllreduceType::AllreduceType)atoi(argw[1]);
    }

    // sizes in Bytes
    std::vector<float> sizes;
    std::vector<int> repetitions;
    switch(allreduce_type){
        case AllreduceType::AllToOne:
        case AllreduceType::Comprex_AllToOne:
            // cannot use big datablocks with AllToOne!
            sizes = std::vector<float>{1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6, 5e6};
            repetitions = std::vector<int>{1000, 1000, 1000, 500, 500, 100, 100, 100, 100, 100, 100, 100};
            break;
        case AllreduceType::Ring:
        case AllreduceType::Comprex_Ring:
            sizes = std::vector<float>{2e3, 5e3, 10e3, 20e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6, 200e6};
            repetitions = std::vector<int>{1000, 1000, 500, 500, 500, 500, 500, 200, 200, 200, 100, 100, 50, 20, 10, 5};
            break;
        case AllreduceType::BigRing:
        case AllreduceType::Comprex_BigRing:
            sizes = std::vector<float>{1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6, 200e6};
            repetitions = std::vector<int>{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 50, 20, 10, 5};
            break;
    }

    // allocation for GASPI segment size
    size_t segment_size = static_cast<size_t>(1) << 31;
    
    if(sizes.size() != repetitions.size()){
        throw std::runtime_error("Size of data-sizes and number of repetitions must match!");
    }
    
    std::vector<float> compression_ratios;
    switch(allreduce_type){
        case AllreduceType::AllToOne:
        case AllreduceType::Ring:
        case AllreduceType::BigRing:
            compression_ratios = std::vector<float>{1.0};
            break;
        case AllreduceType::Comprex_AllToOne:
        case AllreduceType::Comprex_Ring:
        case AllreduceType::Comprex_BigRing:
            compression_ratios = std::vector<float>{0.01, 0.001};
            break;
    }

    // normalize sizes
    for(int i=0; i<sizes.size(); ++i){
        sizes[i] = sizes[i]/sizeof(data_t);
        repetitions[i] = repetitions[i] * 1.0;
    }

    // init GaspiCxx
    GaspiEnvironment gaspi_env(segment_size);

    int myRank =  gaspi_env.get_rank();
    int numRanks = gaspi_env.get_ranks();

    int chiefRank = 0;
    int tag = 0;

    // create a timer
    auto timer = TictocTimer();

    // random data generator
    std::random_device rd{};
    std::mt19937 random_generator{rd()};
    std::normal_distribution<float> normal_distr{0,100};

    // filehandle for result data
    char filename[50];
    switch(allreduce_type){
        case AllreduceType::AllToOne:
            std::cout<<"AllToOne"<<std::endl;
            sprintf(filename, "alltoone_rank%02d.csv", myRank);
            break;
        case AllreduceType::Comprex_AllToOne:
            std::cout<<"Comprex AllToOne"<<std::endl;
            sprintf(filename, "comprexalltoone_rank%02d.csv", myRank);
            break;
        case AllreduceType::Ring:
            std::cout<<"Ring"<<std::endl;
            sprintf(filename, "ring_rank%02d.csv", myRank);
            break;
        case AllreduceType::Comprex_Ring:
            std::cout<<"Comprex Ring"<<std::endl;
            sprintf(filename, "comprexring_rank%02d.csv", myRank);
            break;
        case AllreduceType::BigRing:
            std::cout<<"BigRing"<<std::endl;
            sprintf(filename, "bigring_rank%02d.csv", myRank);
            break;
        case AllreduceType::Comprex_BigRing:
            std::cout<<"Comprex BigRing"<<std::endl;
            sprintf(filename, "comprexbigring_rank%02d.csv", myRank);
            break;
        default:
            throw std::runtime_error("Allreduce type not valid!");
    }
    std::ofstream result_file;
    result_file.open(filename);
    result_file << "size(MB); compression; time_mean(us); time_stddev(us); sparsity_mean; sparsity_stddev\n";

    for(int experiment=0; experiment<sizes.size(); ++experiment){
        for(int compression=0; compression<compression_ratios.size(); ++compression) {
            int size = (int)sizes[experiment];
            int num_runs= repetitions[experiment];
            float compression_ratio = compression_ratios[compression];

            float datasize_MB = (float)size * 1e-6 * sizeof(data_t);
            float size_factor=2.0;

            // setup Allreduce
            Allreduce_base<data_t>* allreduce;
            switch(allreduce_type){
                case AllreduceType::AllToOne:
                    allreduce = new AllToOneAllreduce<data_t>(&gaspi_env);
                    dynamic_cast<AllToOneAllreduce<data_t>*>(allreduce)->setupConnections(size, tag, chiefRank);
                    break;
                case AllreduceType::Comprex_AllToOne:
                    allreduce = new Comprex_AllToOneAllreduce<data_t>(&gaspi_env);
                    dynamic_cast<Comprex_AllToOneAllreduce<data_t>*>(allreduce)->setupConnections(size, tag, chiefRank, compression_ratio, size_factor);
                    break;
                case AllreduceType::Ring:
                    allreduce = new RingAllreduce<data_t>(&gaspi_env);
                    dynamic_cast<RingAllreduce<data_t>*>(allreduce)->setupConnections(size, tag);
                    break;
                case AllreduceType::Comprex_Ring:
                    allreduce = new Comprex_RingAllreduce<data_t>(&gaspi_env);
                    dynamic_cast<Comprex_RingAllreduce<data_t>*>(allreduce)->setupConnections(size, tag, compression_ratio, size_factor);
                    break;
                case AllreduceType::BigRing:
                    allreduce = new BigRingAllreduce<data_t>(&gaspi_env);
                    dynamic_cast<BigRingAllreduce<data_t>*>(allreduce)->setupConnections(size, tag);
                    break;
                case AllreduceType::Comprex_BigRing:
                    allreduce = new Comprex_BigRingAllreduce<data_t>(&gaspi_env);
                    dynamic_cast<Comprex_BigRingAllreduce<data_t>*>(allreduce)->setupConnections(size, tag, compression_ratio, size_factor);
                    break;
                default:
                    throw std::runtime_error("Allreduce type not valid!");
            }

            std::vector<long> measurements(num_runs);
            std::vector<float> meas_sparsity(num_runs);
            
            printf("Allreduce with %.2f MB data and compression ratio %.2f \%\n", datasize_MB, compression_ratio*100);

            // generate test data
            std::vector<data_t> values(size);
            for(int i=0; i<values.size(); ++i) {
                values[i] = std::round(normal_distr(random_generator));
            }

            // results vector
            std::vector<data_t> result(size);
            for(int i=0; i<result.size(); ++i) {
                result[i] = values[i];
            }

            // MAIN LOOP
            // ############################
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
                gaspi_env.barrier();
                timer.tic();
                // do the allreduce
                allreduce->apply(result.data(), size);
                timer.toc();
                measurements[run] = timer.get_time_us();
                meas_sparsity[run] = sparsity(result);

            } //MAIN LOOP
            // #ifdef COMPREX
            //     allreduce.flush(result.data(), size);
            // #endif

            // eval measurements
            float avg_time_us = mean(measurements);
            float stddev_time_us = stddev_mean(measurements);
            float avg_sparsity = mean(meas_sparsity);
            float stddev_sparsity = stddev_mean(meas_sparsity);
            
            printf("Rank %d: size= %.2f MB,  avg. measured time per transaction= %.2f us +/- %f.2 us\n", myRank, datasize_MB, avg_time_us, stddev_time_us);
            result_file << datasize_MB << "; " << compression_ratio << "; " << avg_time_us << "; " << stddev_time_us << "; " << avg_sparsity << "; " << stddev_sparsity << "\n";
            result_file.flush();
            delete allreduce;
        }
    }

    result_file.close();
    gaspi_env.barrier();
    return 0;
}