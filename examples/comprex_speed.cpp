#include <gaspiEx.hxx>
#include <gaspiEx.tpp>
#include <comprEx.hxx>
#include <comprEx.tpp>
#include <threshold.hxx>
#include <compressor.hxx>
#include "tictocTimer.hxx"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <ctime>
#include <experimental/random>
#include <random>

#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/segment/Allocator.hpp>
#include <GaspiCxx/segment/NotificationManager.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
#include <GaspiCxx/utility/ScopedAllocation.hpp>

typedef float data_t;

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

int main(){
    // init GaspiCxx
    gaspi::initGaspiCxx();

    std::vector<float> sizes = {1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6};
    std::vector<int> repetitions = {10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 10, 10};
    if(sizes.size() != repetitions.size()){
        throw std::runtime_error("Size of data-sizes and number of repetitions must match!");
    }
    // normalize sizes
    for(int i=0; i<sizes.size(); ++i){
        sizes[i] = sizes[i]/(sizeof(data_t));
        repetitions[i] = repetitions[i] * 2.0;
    }

    std::vector<float> compression_ratios = { 0.1, 0.01, 0.001};
    
    // init GaspiCxx
    gaspi::segment::Segment segment( static_cast<size_t>(1) << 32);
    gaspi::group::Rank myRank =  gaspi::group::Group().rank();
    gaspi::group::Rank srcRank(0);
    gaspi::group::Rank destRank(1);
    int tag = 1;  // comm tag

    // timer
    auto timer = TictocTimer();

    // random data generator
    std::srand(42);
    std::random_device rd{};
    std::mt19937 random_generator{rd()};
    std::normal_distribution<float> normal_distr{0,100};
    
    // filehandle for result data
    char filename[50];
    sprintf(filename, "results_rank%02d.csv", myRank);
    std::ofstream result_file;
    result_file.open(filename);
    result_file << "size(MB); compression; time_mean(us); time_stddev(us)\n";


    for(int experiment=0; experiment<sizes.size(); ++experiment){
        for(int compression=0; compression<compression_ratios.size(); ++compression) {

            int size=sizes[experiment];
            const int repeat_its=repetitions[experiment];
            float compression_ratio = compression_ratios[compression];

            float datasize_MB = (float)size * 1e-6 * sizeof(data_t);

            if(myRank == srcRank){
                printf("P2P communication with 2X %.2f MB data and compression ratio %.2f %%\n", datasize_MB, compression_ratio*100);
            }

            // setup communicator
            #ifdef COMPREX
                FastComprEx<data_t> communicator(&segment, size, 2);
                communicator.setThreshold(ThresholdTopK<data_t>(compression_ratio));
                // communicator.setThreshold(ThresholdConst<data_t>(9990));
                communicator.setCompressor(CompressorIndexPairs<data_t>());
                // communicator.setCompressor(CompressorRLE<data_t>());
            #else
                GaspiEx<data_t> communicator(&segment, size);
            #endif


            // buffer for sent and received data
            std::vector<data_t> send_values(size);
            std::vector<data_t> recv_values(size);
            for(int i=0; i<send_values.size(); ++i) {
                send_values[i] = std::round(normal_distr(random_generator));
                // send_values[i] = std::rand() %200 -100;
            }

            if(myRank == srcRank){
                communicator.connectTo(destRank, destRank, tag);
            }
            else {
                communicator.connectTo(srcRank, srcRank, tag);
            }

            // vector storing measurements
            std::vector<data_t> measurements(repeat_its);

            int progress=0;
            
            for(int repeat_it=0; repeat_it<repeat_its; ++repeat_it){
                if(myRank == srcRank and repeat_its>=10){
                    if((repeat_it-progress)%(repeat_its/10)==0){
                        std::cout<<((float)repeat_it/repeat_its)*100<<"%..."<<std::endl;
                        progress = repeat_it;
                    }
                }
                
                timer.tic();
                // Source side
                // ------------------------------------------------------------
                if(myRank == srcRank) {
                    // send data to Receiver side.
                    // write test data
                    communicator.writeRemote(send_values.data(), send_values.size());
                    communicator.readRemote_add(recv_values.data(), recv_values.size());
                } 

                // Destination side
                // ------------------------------------------------------------
                if(myRank == destRank) {
                    // get Data from sender
                    communicator.readRemote_add(recv_values.data(), recv_values.size());
                    communicator.writeRemote(send_values.data(), send_values.size());
                    // communicator.readRemote(values.data(), values.size());
                }
                timer.toc();
                measurements[repeat_it]=timer.get_time_us();
            }
            gaspi::getRuntime().barrier();

            // eval measurements
            // bandwidth
            float bytes = 2 * size * sizeof(data_t);
            float avg_time_us = mean(measurements);
            float stddev_time_us = stddev_mean(measurements);
            float bandwidth_MBps = bytes / avg_time_us;
            float bandwidth_uncert = bytes/(avg_time_us*avg_time_us) * stddev_time_us;

            printf("Rank %d: Data= %.2f Bytes, avg. measured time per transaction= %.2f us +/- %f.2 us\n", myRank.get(), bytes, avg_time_us, stddev_time_us);
            printf("Rank %d: Bandwidth= %.2f MBps +/- %.2f MBps\n", myRank.get(), bandwidth_MBps, bandwidth_uncert);

            result_file << datasize_MB << "; " << compression_ratio << "; " << avg_time_us << "; " << stddev_time_us << "\n";
        }
    }

    result_file.close();
    return 0;
}