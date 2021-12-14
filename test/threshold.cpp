#include <threshold.hxx>

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
void print_vector(const std::vector<data_t>& vect){
    std::cout<<"[";
    for(int i=0; i<vect.size(); ++i){
        std::cout<<vect[i]<<", ";
    }
    std::cout<<"\b\b]"<<std::endl;
}

int main(){
    // generate test data
    const int size = 100;
    std::vector<data_t> values(size);
    for(int i=0; i<values.size(); ++i) {
        values[i] = i * (1-2*(i%2));
    }

    // ThresholdConst test
    {
        int errors = 0;
        ThresholdConst<data_t> threshold(20);
        std::cout<<"Test ThresholdConst class."<<std::endl;
        std::cout<<"threshold: "<<threshold.get_threshold()<<std::endl;

        std::vector<data_t> out_vec(size);
        threshold.apply(out_vec.data(), out_vec.size(), values.data(), values.size());

        std::cout<<"Values:"; print_vector<data_t>(values);
        std::cout<<"After Threshold:"; print_vector<data_t>(out_vec);

        for(int i=0; i<values.size(); ++i){
            if( (out_vec[i]!=0) != (std::abs(values[i])>=threshold.get_threshold()) ) {
                std::cout<<values[i]<<">="<<threshold.get_threshold()<<" failed!"<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    }

    // ThresholdTopK test
    {
        int errors = 0;
        ThresholdTopK<data_t> threshold(0.10);
        std::cout<<"Test ThresholdTopK class."<<std::endl;

        std::vector<data_t> out_vec(size);
        threshold.apply(out_vec.data(), out_vec.size(), values.data(), values.size());

        int nelem=0;
        for(auto e : out_vec){
            if(e!=0) ++nelem;
        }

        std::cout<<"threshold: "<<threshold.get_threshold()<<std::endl;
        std::cout<<"#elements: "<<nelem<<" ("<<100.0*(float)nelem/(float)out_vec.size()<<"\%)"<<std::endl;

        std::cout<<"Values:"; print_vector<data_t>(values);
        std::cout<<"After Threshold:"; print_vector<data_t>(out_vec);

        for(int i=0; i<values.size(); ++i){
            if( (out_vec[i]!=0) != (std::abs(values[i])>=threshold.get_threshold()) ) {
                std::cout<<values[i]<<">="<<threshold.get_threshold()<<" failed!"<<std::endl;
                //std::cout<<out_vec[i]<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    }
    


    return 0;
}