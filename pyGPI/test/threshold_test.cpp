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
        std::cout<<"threshold: "<<threshold.getThreshold()<<std::endl;

        std::vector<bool> out_vec_bool;
        std::vector<data_t> out_vec;
        threshold.check(&out_vec_bool, &values);
        threshold.cut(&out_vec, &values);

        std::cout<<"Values:"; print_vector<data_t>(values);
        std::cout<<"After Threshold:"; print_vector<data_t>(out_vec);

        for(int i=0; i<values.size(); ++i){
            if( out_vec_bool[i] != (std::abs(values[i])>=threshold.getThreshold()) ) {
                std::cout<<values[i]<<">="<<threshold.getThreshold()<<" failed!"<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    }

    // ThresholdTopK test
    {
        int errors = 0;
        ThresholdTopK<data_t> threshold(0.42);
        std::cout<<"Test ThresholdTopK class."<<std::endl;

        std::vector<bool> out_vec_bool;
        std::vector<data_t> out_vec;
        threshold.check(&out_vec_bool, &values);
        threshold.cut(&out_vec, &values);

        int nelem=0;
        for(auto e : out_vec_bool){
            if(e) ++nelem;
        }

        std::cout<<"threshold: "<<threshold.getThreshold()<<std::endl;
        std::cout<<"#elements: "<<nelem<<" ("<<100.0*(float)nelem/(float)out_vec.size()<<"\%)"<<std::endl;

        std::cout<<"Values:"; print_vector<data_t>(values);
        std::cout<<"After Threshold:"; print_vector<data_t>(out_vec);

        for(int i=0; i<values.size(); ++i){
            if( out_vec_bool[i] != (std::abs(values[i])>=threshold.getThreshold()) ) {
                std::cout<<values[i]<<">="<<threshold.getThreshold()<<" failed!"<<std::endl;
                //std::cout<<out_vec[i]<<std::endl;
                ++errors;
            }
        }
        print_pass(errors);
    }
    


    return 0;
}