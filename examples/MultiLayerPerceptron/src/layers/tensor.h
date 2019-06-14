#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>
#include <exception>
#include <google/protobuf/repeated_field.h>

template<typename data_t>
class Tensor {
public:
    Tensor<data_t>() {
        this->shape_vec = std::vector<int>();
        this->data_vec = std::vector<data_t>();
    }

    Tensor<data_t>(int* shape, int shape_size) {
        this->shape_vec = std::vector<int>(shape_size);
        for(int i=0; i<shape_size; ++i) {
            this->shape_vec[i]=shape[i];
        }
        this->data_vec = std::vector<data_t>(get_size());
    }

    Tensor<data_t>(std::vector<int> shape) {
        this->shape_vec = std::vector<int>(shape.size());
        for(int i=0; i<shape.size(); ++i) {
            this->shape_vec[i]=shape[i];
        }
        this->data_vec = std::vector<data_t>(get_size());
    }

    Tensor<data_t>(const google::protobuf::RepeatedField<int>& shape) {
        this->shape_vec = std::vector<int>(shape.size());
        for(int i=0; i<shape.size(); ++i) {
            this->shape_vec[i]=shape.Get(i);
        }
        this->data_vec = std::vector<data_t>(get_size());
    }

    template<typename load_t>
    Tensor<data_t>(const google::protobuf::RepeatedField<load_t>& data, const google::protobuf::RepeatedField<int>& shape) {
        this->shape_vec = std::vector<int>(shape.size());
        for(int i=0; i<shape.size(); ++i) {
            this->shape_vec[i]=shape.Get(i);
        }
        this->data_vec = std::vector<data_t>(data.size());
        for(int i=0; i<data.size();++i){
            this->data_vec[i]=data.Get(i);
        }
    }

    Tensor<data_t>(const Tensor<data_t>& rhs){
        shape_vec = rhs.get_shape();
        /*this->shape_vec = std::vector<int>(rhs.get_dims());
        for(int i=0; i<shape_vec.size(); ++i){
            shape_vec[i] = rhs.get_shape(i);
        }*/
        data_vec = rhs.get_data();
        /*data_vec = std::vector<data_t>(rhs.get_size());
        for(int i=0; i<rhs.get_size(); ++i){
            data_vec[i] = rhs.get_data(i);
        }*/
    }

    // try avoiding this call, since it implies the size of the data.
    void load(const data_t* new_data) {
        if(is_empty()){
            throw std::logic_error("Cannot load data into empty Tensor!");
        }
        for(int i=0; i<get_size();++i){
            data_vec[i]=new_data[i];
        }
    }

    template<typename load_t>
    void load(const google::protobuf::RepeatedField<load_t>& data, const google::protobuf::RepeatedField<int>& shape) {
        //this->shape_vec = std::vector<int>(shape.size());
        this->shape_vec.resize(shape.size());
        for(int i=0; i<shape.size(); ++i) {
                shape_vec[i] = shape.Get(i);
        }
        //this->data_vec = std::vector<data_t>(data.size());
        this->data_vec.resize(data.size());
        for(int i=0; i<data.size();++i){
            data_vec[i]=data.Get(i);
        }
    }

    Tensor<data_t>& operator=(const Tensor<data_t>& rhs){
        shape_vec = rhs.get_shape();
        /*shape_vec = std::vector<int>(rhs.get_dims());
        for(int i=0; i<shape_vec.size(); ++i){
            shape_vec[i] = rhs.get_shape(i);
        }*/
        data_vec = rhs.get_data();
        /*//data_vec = std::vector<data_t>(rhs.get_size());
        data_vec.resize(rhs.get_size());
        for(int i=0; i<rhs.get_size(); ++i){
            data_vec[i] = rhs.get_data(i);
        }*/
        return *this;
    }

    Tensor<data_t> operator+(const Tensor<data_t>& rhs) {
        if(this->get_shape() != rhs.get_shape()){
            throw std::runtime_error("Shapes of Tensors are different in addition!");
        }
        Tensor<data_t> lhs(*this);
        for(int i=0; i<lhs.get_size(); ++i){
            lhs[i] += rhs.get_data(i);
        }
        return lhs;
    }

    Tensor<data_t> operator-(const Tensor<data_t>& rhs) {
        if(this->get_shape() != rhs.get_shape()){
            throw std::runtime_error("Shapes of Tensors are different in subtraction!");
        }
        Tensor<data_t> lhs(*this);
        for(int i=0; i<lhs.get_size(); ++i){
            lhs[i] -= rhs.get_data(i);
        }
        return lhs;
    }

    Tensor<data_t> operator*(const data_t& rhs) {
        Tensor<data_t> lhs(*this);
        for(int i=0; i<lhs.get_size(); ++i){
            lhs[i] *= rhs;
        }
        return lhs;
    }

    void reshape(int* shape, int dims){
        shape_vec.resize(dims);
        for(int i=0; i<dims; ++i){
            shape_vec[i] = shape[i];
        }
        data_vec.resize(get_size());
    }

    void reshape(std::vector<int> shape){
        shape_vec = shape;
        data_vec.resize(get_size());
    }

    template<typename load_t>
    void reshape(Tensor<load_t> rhs) {
        reshape(rhs.get_shape());
    }

    void clear() {
        shape_vec.clear();
        data_vec.clear();
    }

    void print_shape() const {
        std::cout<<"[";
        for(int i=0; i<get_dims(); ++i) {
            std::cout<<shape_vec[i]<<", ";
        }
        if(is_empty()) {
            std::cout<<"]"<<std::endl;
        }
        else { 
            std::cout<<"\b\b]"<<std::endl;
        }
    }

    std::string shape_string(){
        std::string shape_str;
        shape_str = "[";
        for(int i=0; i<get_dims(); ++i) {
            shape_str += std::to_string(shape_vec[i]);
            shape_str += ", ";
        }
        shape_str += "\b\b]";
        return shape_str;
    }

    void print(){
        int idx=0;
        print_rec(0, idx);
    }

    int get_size() const {return calc_size(shape_vec);}

    int* shape() {return shape_vec.data();}

    std::vector<int> get_shape() const {return shape_vec;}

    int get_shape(int i) const {return shape_vec[i];}

    int get_dims() const {return shape_vec.size();}

    data_t* data() {return data_vec.data();}

    data_t& operator[](int i) {return data_vec[i];}

    std::vector<data_t> get_data() const {return data_vec;}

    data_t get_data(int i) const {return data_vec[i];}

    bool is_empty() const {return shape_vec.empty();}

    // if tensor effectively has dim of [batch, width] or smaller, it is flat.
    bool is_flat() const {
        for(int i=2; i<get_dims(); ++i){
            if(shape_vec[i] != 1 && shape_vec[i] !=0) return false;
        }
        return true;
    }

private:
    std::vector<int> shape_vec;
    std::vector<data_t> data_vec;

    void print_rec(int depth, int& idx){
        //std::cout<<std::endl;
        for(int j=0; j<depth; ++j) std::cout<<"\t";
        std::cout<<"[ ";
        if(depth==shape_vec.size()-1){
            for(int i=0; i<shape_vec[depth]; ++i) {
                std::cout<<data_vec[idx++]<<", ";
            }
            std::cout<<"\b\b";
        }
        else {
            std::cout<<std::endl;
            for(int i=0; i<shape_vec[depth]; ++i) {
                print_rec(depth+1, idx);
            }
            for(int j=0; j<depth; ++j) std::cout<<"\t";
        }
        std::cout<<"] "<<std::endl;
    }

    int calc_size(std::vector<int> shape) const {
        int s=1;
        for(int i=0; i<shape.size(); ++i){
            s *= shape[i];
        }
        return s;
    }

};




#endif //TENSOR_H
