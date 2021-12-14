#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstring>

/***************************************
 * ThresholdFunction
 * Abstract base class
 ***************************************/ 
template <class VarTYPE>
  class ThresholdFunction{
    public:
        // writes a vector with elements below the threshold set to zero
        virtual void apply(VarTYPE* out_vec, int out_vec_size, const VarTYPE* in_vec, int in_vec_size) {
            if(out_vec_size < in_vec_size){
                throw std::runtime_error("Thresholding target buffer is smaller than source buffer!");
            }
            prepare(in_vec, in_vec_size);
            for(int i=0; i<in_vec_size; ++i){
                out_vec[i] = is_aboveThreshold(in_vec[i]) ? in_vec[i] : static_cast<VarTYPE>(0);
            }
        }

        // writes two vectors, one with the values above the threshold and one with the values below
        virtual void apply_split( VarTYPE* high_vec, int high_vec_size
                                , VarTYPE* low_vec, int low_vec_size
                                , const VarTYPE* in_vec, int in_vec_size) {
            if(high_vec_size < in_vec_size or low_vec_size < in_vec_size){
                throw std::runtime_error("Thresholding target buffer is smaller than source buffer!");
            }            

            prepare(in_vec, in_vec_size);
            VarTYPE threshold=get_threshold();

            for(int i=0; i<in_vec_size; ++i){
                if( std::abs(in_vec[i])>=threshold ){
                    high_vec[i] = in_vec[i];
                    low_vec[i] = static_cast<VarTYPE>(0);
                }
                else {
                    high_vec[i] = static_cast<VarTYPE>(0);
                    low_vec[i] = in_vec[i];
                }
            }
        }

        virtual ThresholdFunction<VarTYPE>* clone() const =0;

        virtual VarTYPE get_threshold()=0;

        // pre comparison calculations, e.g. adjusting threshold
        // optional, does nothing if no override
        virtual void prepare(const VarTYPE* vec, int vec_size) {}

        virtual void prepare_with(const VarTYPE* vec, int vec_size, const VarTYPE* add_vec, int add_vec_size) {}

    protected:
        // element wise comparison function
        virtual bool is_aboveThreshold(const VarTYPE& target)=0;
  };


/***************************************
 * ThresholdNone
 * Apply no threshold, pass everything
 ***************************************/
template <class VarTYPE>
class ThresholdNone : public ThresholdFunction<VarTYPE> {
public:
    virtual ThresholdFunction<VarTYPE>* clone() const {
        return new ThresholdNone<VarTYPE>(*this);
    }

    virtual VarTYPE get_threshold(){
        return 0;
    }

protected:
    virtual bool is_aboveThreshold(const VarTYPE& target) {
        return true;
    }

  };


/***************************************
 * ThresholdConst
 * Constant threshold value
 * compares |value| >= threshold
 ***************************************/
template <class VarTYPE>
class ThresholdConst : public ThresholdFunction<VarTYPE> {
public:
    ThresholdConst(VarTYPE threshold) : _threshold(threshold) {
    if(_threshold<0){ throw std::runtime_error("Threshold of ThresholdConst is <0!"); }
    }

    virtual ThresholdFunction<VarTYPE>* clone() const {
    return new ThresholdConst<VarTYPE>(*this);
    }

    virtual VarTYPE get_threshold(){
        return _threshold;
    }

protected:
    VarTYPE _threshold;

    virtual bool is_aboveThreshold(const VarTYPE& target){
        if(std::abs(target) >= _threshold) return true;
        else return false;
    }

  }; 


/***************************************
 * ThresholdTopK
 * adaptive threshold value, giving back the top k% of vector
 * compares |value| >= threshold, where threshold is computed automatically after calling `prepare()`.
 ***************************************/
template <class VarTYPE>
class ThresholdTopK : public ThresholdFunction<VarTYPE> {
public:
    ThresholdTopK(float topK, int default_subsamples=1000, float subsample_factor=10.0) : _threshold(0) {
        if(topK > 1.0 || topK<0.0){
            throw std::runtime_error("topK value must be between 0.0 and 1.0!");
        }
        this->topK = topK;
        // minimum number of samples the threshold should be estimated on.
        this->default_subsamples = default_subsamples;
        // Make sure that the number of samples is at least (subsample_factor / top_k).
        this->subsamples_factor = subsample_factor;
    }

    float getTopK() const { return topK; }

    virtual ThresholdFunction<VarTYPE>* clone() const {
        return new ThresholdTopK<VarTYPE>(*this);
    }

    virtual VarTYPE get_threshold(){
        return _threshold;
    }

    // Calculate new threshold for given top_k value.
    virtual void prepare(const VarTYPE* vec, int vec_size) {
        std::vector<VarTYPE> vec_copy;
        prep_vec(vec_copy, vec, vec_size);
        _threshold = calc_threshold(vec_copy);
    }

    // Calculate new threshold for given top_k value, considering that a second vector will be added.
    // only adds elements from `add_vec` to the subsampled set.
    virtual void prepare_with(const VarTYPE* vec, int vec_size, const VarTYPE* add_vec, int add_vec_size) {
        if(vec_size != add_vec_size){
              throw std::runtime_error("vector sizes must be the same!");
        }
        std::vector<VarTYPE> vec_copy;
        prep_add_vec(vec_copy, vec, add_vec, vec_size);
        _threshold = calc_threshold(vec_copy);
    }

protected:
    float topK;
    int subsamples;
    int default_subsamples;
    float subsamples_factor;
    VarTYPE _threshold;

    virtual bool is_aboveThreshold(const VarTYPE& target){
      if( std::abs(target) >= _threshold ) return true;
      else return false;
    }

    // Calculates the number of samples which will be used to calculate the threshold.
    int calc_subsamples(){
        float subsamples_req = subsamples_factor/topK;
        int subsamples = (subsamples_req > default_subsamples) ? subsamples_req : default_subsamples;
        return subsamples;
    }

    // Calculate the threshold based on the top_k value and the (subsampled) vector `vec_copy`.
    VarTYPE calc_threshold(std::vector<VarTYPE>& vec_copy){
        int topk_pos = std::ceil(vec_copy.size() * topK)-1;
        VarTYPE new_threshold;

        if(topk_pos<=0){
            VarTYPE maximum = *std::max_element( vec_copy.begin(), vec_copy.end(), [](VarTYPE x1, VarTYPE x2){return std::abs(x1) < std::abs(x2);} );
            new_threshold = std::abs(maximum)+1;
        }
        else {
            // std::sort(vec_copy.begin(), vec_copy.end(), [](VarTYPE x1,VarTYPE x2){return std::abs(x1) > std::abs(x2);} );
            std::nth_element (vec_copy.begin(), vec_copy.begin()+topk_pos, vec_copy.end(), [](VarTYPE x1,VarTYPE x2){return std::abs(x1) > std::abs(x2);});
            new_threshold = std::abs((vec_copy)[topk_pos]);
            // VarTYPE minimum = *std::min_element( vec_copy.begin(), vec_copy.end(), [](VarTYPE x1, VarTYPE x2){return std::abs(x1) < std::abs(x2);} );
            // _threshold = std::abs(minimum);
        } 
        return new_threshold;
    }

    // subsamples from `vec` into `vec_copy`.
    void prep_vec(std::vector<VarTYPE>& vec_copy, const VarTYPE* vec, int vec_size){
        int subsamples = calc_subsamples();
        if(vec_size<subsamples){
            vec_copy = std::vector<VarTYPE>(vec, vec+vec_size);
        }
        else {
            vec_copy.resize(subsamples);
            float stepsize = (float)vec_size/(float)subsamples;
            for(int i=0; i<subsamples; ++i){
                int steppos = (int)(i*stepsize);
                vec_copy[i] = vec[steppos];
            }
        }
    }

    // subsamples `vec` into `vec_copy` and adds corresponding values from `add_vec`.
    void prep_add_vec(std::vector<VarTYPE>& vec_copy, const VarTYPE* vec, const VarTYPE* add_vec, int vec_size){
        int subsamples = calc_subsamples();
        if(vec_size<subsamples){
            vec_copy.resize(vec_size);
            for(int i=0; i<vec_size; ++i){
                vec_copy[i] = vec[i] + add_vec[i];
            }
        }
        else {
            vec_copy.resize(subsamples);
            float stepsize = (float)vec_size/(float)subsamples;
            for(int i=0; i<subsamples; ++i){
                int steppos = (int)(i*stepsize);
                vec_copy[i] = vec[steppos] + add_vec[steppos];
            }
        }
    }

  };

#endif //THRESHOLD_H
