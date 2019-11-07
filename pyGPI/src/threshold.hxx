#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>

/***************************************
 * ThresholdFunction
 * Abstract base class
 ***************************************/ 
template <class VarTYPE>
  class ThresholdFunction{
    public:
        // writes a vector with elements below the threshold set to zero
        virtual void cut(std::vector<VarTYPE>* out_vec, const std::vector<VarTYPE>* in_vec) {
            out_vec->resize(in_vec->size());
            prepare(in_vec);
            for(int i=0; i<out_vec->size(); ++i){
                (*out_vec)[i] = (is_aboveThreshold((*in_vec)[i])) ? (*in_vec)[i] : static_cast<VarTYPE>(0);
            }
        }

        // writes a boolean mask for the values
        virtual void check(std::vector<bool>* out_vec, const std::vector<VarTYPE>* in_vec) {
            out_vec->resize(in_vec->size());
            prepare(in_vec);
            for(int i=0; i<out_vec->size(); ++i){
                (*out_vec)[i] = ( is_aboveThreshold((*in_vec)[i]) );
            }
        }

        virtual std::unique_ptr<ThresholdFunction<VarTYPE> > copy() const =0;

    protected:
        // element wise comparison function
        virtual bool is_aboveThreshold(const VarTYPE& target)=0;

        // pre comparison calculations, e.g. adjusting threshold
        // optional, does nothing if no override
        virtual void prepare(const std::vector<VarTYPE>* vec) {}
  };


/***************************************
 * ThresholdNone
 * Apply no threshold, pass everything
 ***************************************/
template <class VarTYPE>
class ThresholdNone : public ThresholdFunction<VarTYPE> {
public:
    virtual std::unique_ptr<ThresholdFunction<VarTYPE> > copy() const {
        return std::unique_ptr<ThresholdFunction<VarTYPE> >{new ThresholdNone<VarTYPE>(*this)};
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

      VarTYPE getThreshold(){ return _threshold; }

      virtual std::unique_ptr<ThresholdFunction<VarTYPE> > copy() const {
        return std::unique_ptr<ThresholdFunction<VarTYPE> >{new ThresholdConst<VarTYPE>(*this)};
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
 * compares |value| >= threshold, where threshold is computed automatically
 ***************************************/
template <class VarTYPE>
class ThresholdTopK : public ThresholdFunction<VarTYPE> {
public:
    ThresholdTopK(float topK) : _threshold(0) {
          if(topK > 1.0 || topK<0.0){
              throw std::runtime_error("topK value must be between 0.0 and 1.0!");
          }
          _topK = topK;
    }

    float getTopK() const { return _topK; }

    VarTYPE getThreshold() const { return _threshold; }

    virtual std::unique_ptr<ThresholdFunction<VarTYPE> > copy() const {
        return std::unique_ptr<ThresholdFunction<VarTYPE> >{new ThresholdTopK<VarTYPE>(*this)};
    }

protected:
    float _topK;
    VarTYPE _threshold;

    virtual bool is_aboveThreshold(const VarTYPE& target){
      if( std::abs(target) >= _threshold ) return true;
      else return false;
    }

    virtual void prepare(const std::vector<VarTYPE>* vec) {
        std::vector<VarTYPE> vec_copy(*vec);
        int topk_pos = std::ceil(vec_copy.size() * _topK)-1;
        // sort vector in descending order
        std::sort(vec_copy.begin(), vec_copy.end(), [](VarTYPE x1,VarTYPE x2){return std::abs(x1) > std::abs(x2);} );
        _threshold = (topk_pos<0) ? std::abs((vec_copy)[0])+1 : std::abs((vec_copy)[topk_pos]);
    }
  };


#endif //THRESHOLD_H