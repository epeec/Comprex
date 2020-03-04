#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include<cassert>
//#define NDEBUG

#include<stdexcept>
#include<vector>
#include <memory>

#include<compressedVector.hxx>

/***************************************
 * Compressor
 * Abstract base class
 * assume that input is after threshold, i.e. sparse.
 ***************************************/
template <class VarTYPE>
class Compressor{
    public:
        // compression, returns compressed vector
        virtual void compress(std::unique_ptr<CompressedVector<VarTYPE> >* output, const std::vector<VarTYPE>* input){
            compression_strategy(output, input);
        }

        // compression, writes compressed vector into buffer
        virtual void compress(void* targetBuffer, const std::vector<VarTYPE>* input){
            std::unique_ptr<CompressedVector<VarTYPE> > cVect;
            compression_strategy(&cVect, input);
            cVect.get()->writeBuffer(targetBuffer);
        }

        // decompress vector from compressed Vector
        virtual void decompress(std::vector<VarTYPE>* output, const CompressedVector<VarTYPE>* input) {
            output->resize(input->uncompressed_size());
            decompression_strategy(output->data(), output->size(), input);
        }

        // decompress vector from compressed Vector into buffer
        virtual void decompress(VarTYPE* output, int size, const CompressedVector<VarTYPE>* input) {
            if(input->uncompressed_size() != size){
                char error_msg[100];
                sprintf(error_msg,"Decompressing Vector of size %d does not match the buffer of size %d!\n", (int)input->uncompressed_size(), (int)size);
                throw std::runtime_error(error_msg);
            }
            decompression_strategy(output, size, input);
        }

        // decompress vector from buffer
        virtual void decompress(std::vector<VarTYPE>* output, void* sourceBuffer){
            auto cVect = getEmptyCompressedVector();
            cVect.get()->loadBuffer(sourceBuffer);
            output->resize(cVect.get()->uncompressed_size());
            decompression_strategy(output->data(), output->size(), cVect.get());
        }

        // get an instance of the fitting Compressed Vector type
        virtual std::unique_ptr<CompressedVector<VarTYPE> > getEmptyCompressedVector()=0;

        // returns a copy of this instance
        virtual std::unique_ptr<Compressor<VarTYPE> > copy() const =0;

    protected:
        // the implementation of the compression strategy.
        // Gets a reference to a vector of values and must return a unique_ptr to a compressed Vector.
        virtual void compression_strategy(std::unique_ptr<CompressedVector<VarTYPE> >* output, const std::vector<VarTYPE>* input) = 0;

        virtual void decompression_strategy(VarTYPE* output, int size, const CompressedVector<VarTYPE>* input) = 0;

        // calculate size based on boolean mask
        int calcSize(const std::vector<bool>& vec){
            int size=0;
            for(auto v : vec) if(v) ++size;
            return size;
        }
        // calculate size based on sparse vector
        int calcSize(const std::vector<VarTYPE>& vec){
            int size=0;
            for(auto v : vec) if( v != static_cast<VarTYPE>(0) ) ++size;
            return size;
        }

}; //class Compressor


/***************************************
 * CompressorMultiThreaded
 * Abstact base class for multi threaded compressors
 ***************************************/
template <class VarTYPE>
class CompressorMultiThreaded : public Compressor<VarTYPE> {
public:

    CompressorMultiThreaded() : _nThreads(1) {}

    CompressorMultiThreaded(int nThreads, const std::vector<int>& pinPattern) 
            : _nThreads(nThreads), _pinPattern(pinPattern) {
        // multithreading
        if(_nThreads > 1){
            // partition input vector
            _startIdx_thr = std::vector<int>(_nThreads+1); //std::unique_ptr<int []> (new int [_numThreads+1]);
            _signumFlag_thr = std::vector<int>(_nThreads); //std::unique_ptr<int []> (new int [_numThreads]);
            //_runLengthsVectSize_thr = std::vector<int>(_nThreads); //std::unique_ptr<int []> (new int [_numThreads]);
            //_compressedVectSize_thr = std::vector<int>(_nThreads); // std::unique_ptr<int []> (new int [_numThreads]);
            _runLengthsVect_thr = std::unique_ptr< std::vector<int> []> 
                                            (new std::vector<int> [_nThreads]);
            _compressedVect_thr = std::unique_ptr< std::vector<VarTYPE> []> 
                                            (new std::vector<VarTYPE> [_nThreads]);
                    
        }
    }

    virtual void setPinPattern( const std::vector<int>& pinPattern ) {
        if(pinPattern.size() != _nThreads) {
            throw std::runtime_error("New pinPattern does not match size of nThreads!");
        }
        _pinPattern = pinPattern;
    }

protected:
    // pin_pattern, number of threads, in case of a multi-threaded compression
    // TODO: part of compression strategy
    int _nThreads;
    std::vector<int> _pinPattern;

    std::vector<int> _startIdx_thr;
    std::vector<int> _signumFlag_thr;
    std::unique_ptr< std::vector<int> []> _runLengthsVect_thr;
    std::unique_ptr< std::vector<VarTYPE> []> _compressedVect_thr;

    void partitionThreads(int input_size) {
        this->_startIdx_thr[0] = 0;
        int size = input_size /_nThreads;
        for(int i = 1; i < _nThreads; i++) {
            _startIdx_thr[i] = _startIdx_thr[i-1]+size;
            if(i < input_size % _nThreads ) ++_startIdx_thr[i];
        }
        _startIdx_thr[_nThreads] = input_size;
    }
}; //class Compressor


/***************************************
 * CompressorNone
 * Apply no compression to input data
 ***************************************/
template <class VarTYPE>
class CompressorNone : public Compressor<VarTYPE> {
public:
    CompressorNone<VarTYPE>(){}

    virtual std::unique_ptr<CompressedVector<VarTYPE> > getEmptyCompressedVector(){
        return std::unique_ptr<CompressedVector<VarTYPE> >{new CompressedVectorNone<VarTYPE>()};
    }

    virtual std::unique_ptr<Compressor<VarTYPE> > copy() const {
        return std::unique_ptr<Compressor<VarTYPE> >{new CompressorNone<VarTYPE>(*this)};
    }

protected:
    virtual void compression_strategy(std::unique_ptr<CompressedVector<VarTYPE> >* output, const std::vector<VarTYPE>* input) {
        *output = std::unique_ptr<CompressedVector<VarTYPE> >{new CompressedVectorNone<VarTYPE>(input)};
    }

    virtual void decompression_strategy(VarTYPE* output, int size, const CompressedVector<VarTYPE>* input) {
        std::vector<VarTYPE> vect = dynamic_cast<const CompressedVectorNone<VarTYPE>*>(input)->getVect();
        for(int i=0; i<size; ++i){
            output[i] = vect[i];
        }
    }
}; //class Compressor


/***************************************
 * CompressorRLE
 * Run length encoding
 ***************************************/
template<typename VarTYPE>
class CompressorRLE : public CompressorMultiThreaded<VarTYPE> {
public:
    CompressorRLE() : CompressorMultiThreaded<VarTYPE>() {
    }

    CompressorRLE(const CompressorRLE<VarTYPE>& rhs) 
            : CompressorMultiThreaded<VarTYPE>( rhs._nThreads, rhs._pinPattern)
             {
    }

    CompressorRLE(int nThreads, const std::vector<int>& pinPattern)
                    : CompressorMultiThreaded<VarTYPE>(nThreads, pinPattern) {
    }

    virtual std::unique_ptr<CompressedVector<VarTYPE> > getEmptyCompressedVector(){
            return std::unique_ptr<CompressedVector<VarTYPE> >{new CompressedVectorRLE<VarTYPE>()};
    }

    virtual std::unique_ptr<Compressor<VarTYPE> > copy() const {
        return std::unique_ptr<Compressor<VarTYPE> >{new CompressorRLE<VarTYPE>(*this)};
    }
    
protected:
    virtual void compression_strategy(std::unique_ptr<CompressedVector<VarTYPE> >* output, const std::vector<VarTYPE>* input) {
        // uncompressed Vector
        int original_size = input->size();
        // compressed Vector
        int compressed_size = Compressor<VarTYPE>::calcSize(*input);
        std::vector<VarTYPE> compressedVector(compressed_size);
        int compressedVectorPos = 0; // counter for output vector position

        // compression statistics
        int check_cntr = 0; // increase it each time _auxInfoVect.push_back(..) is executed
        int crrRunLength_yes = 0;
        int crrRunLength_no = 0;

        // Auxilliary Vector, for decompression
        int signum = 0; // binary value, but leave it int, because it needs to be sent
        std::vector<int> auxInfoVect;

        // check the first vector item, set the signum-flag
        // -------------------------------------------------------
        if((*input)[0] != static_cast<VarTYPE>(0)) {  
            //compressedVector.push_back(input[0]); // dynamic
            compressedVector[compressedVectorPos++] = (*input)[0];

            crrRunLength_yes++;    // increase the "yes"-counter   
            signum = 1;
        }
        else {  //if(_restsVect[0] < treshold)
            // increase the  crrRunLength_no-counter and set the signum
            crrRunLength_no++;
            signum = 0;
        }

        // Splitting the zeros from the original array
        // -------------------------------------------------------
        for(int i = 1; i < input->size(); i++) {
            if( (*input)[i] != static_cast<VarTYPE>(0) ) { // add element to compressed vector
         
                compressedVector[compressedVectorPos++] = (*input)[i];

                crrRunLength_yes++;

                if(crrRunLength_no > 0) {  //the previous number was a "no"-number 
                    // write the current "no"-length in _auxInfoVect[] 
                    // and increase the counter 
                    auxInfoVect.push_back(crrRunLength_no);
                    //increase the check-counter (evntl. optional)
                    check_cntr += crrRunLength_no;
                    // set crrRunLength_no to ZERO
                    crrRunLength_no = 0;
                }
            }
            else {  // skip zeros
                crrRunLength_no++;         // Then just increase the counter
                if(crrRunLength_yes > 0) {//the previous number was an "yes"-number 
                    // write the current "yes"-length in _auxInfoVectr[] 
                    // (it should be != 0) and increase the counter 
                    auxInfoVect.push_back(crrRunLength_yes); 
                    //increase the check-counter (evntl. optional)
                    check_cntr += crrRunLength_yes;
                    // set crrRunLength_yes to ZERO
                    crrRunLength_yes=0;
                }
            }
        }

        // push_back the last (still not stored) sequence
        if( (crrRunLength_yes > 0) && (crrRunLength_no == 0)) {
            auxInfoVect.push_back(crrRunLength_yes); 
            //increase the check-counter (evntl. optional)
            check_cntr += crrRunLength_yes;
        }
        if( (crrRunLength_no > 0) && (crrRunLength_yes == 0)) {
            auxInfoVect.push_back(crrRunLength_no); 
            //increase the check-counter (evntl. optional)
            check_cntr += crrRunLength_no;
        }
        
        // check here if the sum of _auxInfoVectr[]-items 
        // equals the original size, if not -> throw
        assert(check_cntr != input->size());

        *output = std::unique_ptr<CompressedVectorRLE<VarTYPE> >{ new CompressedVectorRLE<VarTYPE>(original_size, signum, auxInfoVect, compressedVector)};
    }


    virtual void decompression_strategy(VarTYPE* output, int size, const CompressedVector<VarTYPE>* input) {
        // compressed vector
        const CompressedVectorRLE<VarTYPE>* input_RLE_p = dynamic_cast<const CompressedVectorRLE<VarTYPE>*>(input);
        int signum = input_RLE_p->get_signum();
        int original_size = input_RLE_p->get_original_size();
        const std::vector<int>* auxInfoVect_p = input_RLE_p->get_auxInfoVect_p();
        const std::vector<VarTYPE>* compressedVect_p = input_RLE_p->get_compressedVect_p();

        // decompression statistics
        int cntr_orig = 0;
        int cntr_comprs = 0;
        int cntr_runLengths = 0;

        if(signum == 0) {  // strting with "no"-sequence
            cntr_orig += (*auxInfoVect_p)[cntr_runLengths];
            for(int i = 0; i < cntr_orig; i++) {
                (output)[i] = 0;
            }
            cntr_runLengths++;
        }

        // now alternate (i) "yes"-items (ii) "no" items
        while (cntr_orig < original_size) {
            // "yes"-items
            for(int i = cntr_orig; i < cntr_orig + (*auxInfoVect_p)[cntr_runLengths]; i++) {
                (output)[i] = (*compressedVect_p)[cntr_comprs];
                cntr_comprs++;
            }
            cntr_orig += (*auxInfoVect_p)[cntr_runLengths];
            cntr_runLengths++;

            if(cntr_orig == original_size) break;
            // "no"-items
            for(int i = cntr_orig; i < cntr_orig + (*auxInfoVect_p)[cntr_runLengths]; i++) {
                (output)[i] = 0;       
            }
            cntr_orig += (*auxInfoVect_p)[cntr_runLengths];
            cntr_runLengths++;
        }
    }

};

#endif //COMPRESSOR_H