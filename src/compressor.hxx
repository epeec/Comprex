#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include<cassert>
//#define NDEBUG

#include<stdexcept>
#include<vector>
#include<memory>
#include<cmath>
#include<algorithm>
#include<cstring>

#include "threshold.hxx"


// #include<compressedVector.hxx>

/***************************************
 * Compressor
 * Abstract base class
 * assume that input is after threshold, i.e. sparse.
 ***************************************/
template <class VarTYPE>
class Compressor{
    public:

        Compressor(){}

        Compressor(const Compressor<VarTYPE>& other){}

        virtual ~Compressor(){}

        // compression, writes compressed vector into buffer
        // returns how many bytes were written to targetBuffer
        virtual int compress(void* targetBuffer, const VarTYPE* data_vector, const int data_vector_size, VarTYPE threshold){
            return compression_strategy(targetBuffer, data_vector, NULL, data_vector_size, threshold);
        }
        // former compress_with_threshold
        virtual int compress_update(void* targetBuffer, VarTYPE* update_vector, const int update_vector_size, VarTYPE threshold){
            return compression_strategy(targetBuffer, NULL, update_vector, update_vector_size, threshold);
        }
        // former compress_add_with_threshold
        virtual int compress_add_update(void* targetBuffer, VarTYPE* update_vector, const int update_vector_size, const VarTYPE* data_vector, const int data_vector_size, VarTYPE threshold){
            if(data_vector_size != update_vector_size){
                char msg[100];
                sprintf(msg, "Compressor: Update vector size (%d) is not equal data vector size (%d)!", update_vector_size, data_vector_size );
                throw std::runtime_error(msg);
            }
            return compression_strategy(targetBuffer, data_vector, update_vector, update_vector_size, threshold);
        }

        // decompress vector from buffer
        // returns how long (#elements) the read vector is
        virtual int  decompress(VarTYPE* output, int output_size, const void* sourceBuffer){
            return decompression_strategy(output, output_size, sourceBuffer);
        }

        // add a buffer to a vector, fuses decompression with add operator
        virtual void add(VarTYPE* output, int output_size, const void* sourceBuffer){
            return add_strategy(output, output_size, sourceBuffer);
        }

        // returns a copy of this instance
        virtual Compressor<VarTYPE>* clone() const =0;

    protected:
        // the implementation of the compression strategy.
        // Gets a reference to a vector of values and must return a unique_ptr to a compressed Vector.
        virtual int compression_strategy(void* output, const VarTYPE* data_vector, VarTYPE* update_vector, const int vector_size, VarTYPE threshold) = 0;

        virtual int decompression_strategy(VarTYPE* output, int output_size, const void* input) = 0;

        virtual void add_strategy(VarTYPE* output, int output_size, const void* sourceBuffer) = 0;

}; //class Compressor


/***************************************
 * CompressorNone
 * Apply no compression to input data
 ***************************************/
template <class VarTYPE>
class CompressorNone : public Compressor<VarTYPE> {
public:
    CompressorNone<VarTYPE>(){}

    virtual Compressor<VarTYPE>* clone() const {
        return new CompressorNone<VarTYPE>(*this);
    }

protected:
    virtual int compression_strategy(void* output, const VarTYPE* data_vector, VarTYPE* update_vector, const int vector_size, VarTYPE threshold) {
        void* pCrr = output;
        // write size of vector
        *(reinterpret_cast<int *>(pCrr)) = vector_size;
        pCrr += sizeof(int);
        // write data
        if(update_vector != NULL && data_vector != NULL){
            VarTYPE* p_buffer = reinterpret_cast<VarTYPE*>(pCrr);
            for(int i=0; i<vector_size; ++i){
                p_buffer[i] = data_vector[i] + update_vector[i];
                update_vector[i] = 0;
            }
        }
        else if(update_vector != NULL) {
            std::memcpy( pCrr, update_vector, vector_size*sizeof(VarTYPE) );
            std::memset( update_vector, 0, vector_size*sizeof(VarTYPE));
        }
        else if(data_vector != NULL) {
            std::memcpy( pCrr, data_vector, vector_size*sizeof(VarTYPE) );
        }
        else{
            throw std::runtime_error("Compression needs a valid input vector!");
        }
        pCrr +=  vector_size*sizeof(VarTYPE);
        return (char const volatile*)pCrr-(char const volatile*)output;
    }

    virtual int decompression_strategy(VarTYPE* output, int output_size, const void* input) {
        const void* pCrr = input;
        // load and adjust size of vector
        int r_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        if (r_size>output_size){
            throw std::runtime_error("Decompression vector size is bigger than output buffer!");
        }
        // load data into vector
        std::memcpy(output, pCrr, r_size*sizeof(VarTYPE) ); 
        pCrr +=  r_size*sizeof(VarTYPE);
        int sizeBytes = (char const volatile*)pCrr-(char const volatile*)input;
        //return r_size;
        return sizeBytes;
    }

    virtual void add_strategy(VarTYPE* output, int output_size, const void* sourceBuffer){
        const void* pCrr = sourceBuffer;
        // load and adjust size of vector
        int r_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        if (r_size>output_size){
            throw std::runtime_error("Decompression vector size is bigger than output buffer!");
        }
        const VarTYPE* data = reinterpret_cast<const VarTYPE *>(pCrr);
        pCrr +=  r_size*sizeof(VarTYPE);
        // add data to vector
        for(int it = 0; it<output_size; ++it){
            output[it] += data[it];
        }
    }
}; //class Compressor


/***************************************
 * CompressorRLE
 * Run length encoding
 ***************************************/
template<typename VarTYPE>
class CompressorRLE : public Compressor<VarTYPE> {
public:
    CompressorRLE() {
    }

    virtual Compressor<VarTYPE>* clone() const {
        return new CompressorRLE<VarTYPE>(*this);
    }
    
protected:
    virtual int compression_strategy(void* output, const VarTYPE* data_vector, VarTYPE* update_vector, const int vector_size, VarTYPE threshold) {
        // uncompressed Vector
        int original_size = vector_size;
        // compressed Vector
        int compressed_size = 0;

        // compression statistics
        int check_cntr = 0; // increase it each time _auxInfoVect.push_back(..) is executed
        int crrRunLength_yes = 0;
        int crrRunLength_no = 0;

        // Auxilliary Vector, for decompression
        int signum = 0; // binary value, but leave it int, because it needs to be sent
        std::vector<int> auxInfoVect;

        void* pCrr = output;

        // originalVectorSize , int
        *(reinterpret_cast<int *>(pCrr)) = original_size;
        pCrr += sizeof(int);

        // shrinkedVectorSize , write later
        int* pShrinkedVectorSize = reinterpret_cast<int *>(pCrr);
        pCrr += sizeof(int);

        // shrinkedVector, write it while estimating auxInfoVect
        // check the first vector item, set the signum-flag
        // -------------------------------------------------------
        VarTYPE* p_buffer = reinterpret_cast<VarTYPE *>(pCrr);
        if(update_vector != NULL && data_vector != NULL) {
            update_vector[0] += data_vector[0];
        }

        const VarTYPE* vector;
        if(update_vector != NULL){
            vector = update_vector;
        }
        else if(data_vector != NULL){
            vector = data_vector;
        }
        else {
            throw std::runtime_error("Compression needs a valid input vector!");
        }
        if(std::abs(vector[0]) >= threshold) {
            p_buffer[0] = vector[0];
            // erase written element from buffer
            if(update_vector != NULL) {
                update_vector[0] = 0;
            }
            ++compressed_size;
            pCrr += sizeof(VarTYPE);


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
        if(update_vector != NULL && data_vector != NULL){
            for(int i = 1; i < vector_size; i++) {
                update_vector[i] += data_vector[i];
                if( std::abs(update_vector[i]) >= threshold ) { // add element to compressed vector
                    //_shrinkedVect[_shrinkedSize] = _restsVector[i];
                    p_buffer[i] = update_vector[i];
                    // erase written element from buffer
                    update_vector[i] = 0;
                    ++compressed_size;
                    pCrr += sizeof(VarTYPE);

                    //if(crrRunLength_yes >= 0) { // the previous number was an "yes"-number
                    crrRunLength_yes++; // Then just increase the counter
                    //}
                    if(crrRunLength_no > 0) {  //the previous number was a "no"-number 
                    // write the current "no"-length in _auxInfoVect[] 
                    // and increase the counter 
                    auxInfoVect.push_back(crrRunLength_no);
                    //increase the check-counter (evntl. optional)
                    check_cntr += crrRunLength_no;
                    // set crrRunLength_no to ZERO
                    crrRunLength_no = 0;
                    } // if(crrRunLength_no > 0)
                }
                else {  // skip zeros
                    //if(crrRunLength_no >= 0) {    // the previous number was a "no"-number
                        crrRunLength_no++;         // Then just increase the counter
                    //}
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
            } //for(int i;...)
        }
         else if(update_vector != NULL){
            for(int i = 1; i < vector_size; i++) {
                if( std::abs(update_vector[i]) >= threshold ) { // add element to compressed vector
                    //_shrinkedVect[_shrinkedSize] = _restsVector[i];
                    p_buffer[i] = update_vector[i];
                    // erase written element from buffer
                    update_vector[i] = 0;
                    ++compressed_size;
                    pCrr += sizeof(VarTYPE);

                    //if(crrRunLength_yes >= 0) { // the previous number was an "yes"-number
                    crrRunLength_yes++; // Then just increase the counter
                    //}
                    if(crrRunLength_no > 0) {  //the previous number was a "no"-number 
                    // write the current "no"-length in _auxInfoVect[] 
                    // and increase the counter 
                    auxInfoVect.push_back(crrRunLength_no);
                    //increase the check-counter (evntl. optional)
                    check_cntr += crrRunLength_no;
                    // set crrRunLength_no to ZERO
                    crrRunLength_no = 0;
                    } // if(crrRunLength_no > 0)
                }
                else {  // skip zeros
                    //if(crrRunLength_no >= 0) {    // the previous number was a "no"-number
                        crrRunLength_no++;         // Then just increase the counter
                    //}
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
            } //for(int i;...)
        }
        else if(data_vector != NULL){
            for(int i = 1; i < vector_size; i++) {
                if( std::abs(data_vector[i]) >= threshold ) { // add element to compressed vector
                    //_shrinkedVect[_shrinkedSize] = _restsVector[i];
                    p_buffer[i] = data_vector[i];
                    ++compressed_size;
                    pCrr += sizeof(VarTYPE);

                    //if(crrRunLength_yes >= 0) { // the previous number was an "yes"-number
                    crrRunLength_yes++; // Then just increase the counter
                    //}
                    if(crrRunLength_no > 0) {  //the previous number was a "no"-number 
                    // write the current "no"-length in _auxInfoVect[] 
                    // and increase the counter 
                    auxInfoVect.push_back(crrRunLength_no);
                    //increase the check-counter (evntl. optional)
                    check_cntr += crrRunLength_no;
                    // set crrRunLength_no to ZERO
                    crrRunLength_no = 0;
                    } // if(crrRunLength_no > 0)
                }
                else {  // skip zeros
                    //if(crrRunLength_no >= 0) {    // the previous number was a "no"-number
                        crrRunLength_no++;         // Then just increase the counter
                    //}
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
            } //for(int i;...)
        }
        else {
            throw std::runtime_error("Compression needs a valid input vector!");
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
        // May leave this check, it is not "expensive"
        assert(check_cntr != original_size);

        //  run-Length-vector Size, int
        *(reinterpret_cast<int *>(pCrr)) = auxInfoVect.size();
        pCrr += sizeof(int);
    
        // signumFlag, int
        *(reinterpret_cast<int *>(pCrr)) = signum;
        pCrr += sizeof(int);

        // _runLengthsSize integers, the contents of _auxInfoVect[]-vector
        std::memcpy( pCrr, auxInfoVect.data(), auxInfoVect.size()*sizeof(int) ); 
        pCrr +=  auxInfoVect.size()*sizeof(int);
        
        *pShrinkedVectorSize = compressed_size;

        return (char const volatile*)pCrr-(char const volatile*)output;
    } // compression_strategy


    virtual int decompression_strategy(VarTYPE* output, int output_size, const void* input) {

        const void* pCrr = input;
        const VarTYPE* pShrinkedVector;

        // received data statistics
        std::vector<int> auxInfoVect;
        int r_original_size=0;
        int r_compressed_size=0;
        int r_auxInfoVect_size=0;
        int r_signum=0;

        // originalVectorSize , int
        r_original_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        if(r_original_size>output_size){
            throw std::runtime_error("Decompression vector size is bigger than output buffer!");
        }
        
        // shrinkedVectorSize , int
        r_compressed_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);

        // _shrinkedSize VarTYPE, decompress later from this position
        pShrinkedVector = reinterpret_cast<const VarTYPE *>(pCrr);
        pCrr += r_compressed_size*sizeof(VarTYPE);

        //  run-Length-vector Size, int
        r_auxInfoVect_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        auxInfoVect.resize(r_auxInfoVect_size);
    
        // signumFlag, int
        r_signum = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        
        // _runLengthsSize integers, the contents of _auxInfoVect[]-vector
        std::memcpy(auxInfoVect.data(), pCrr, auxInfoVect.size()*sizeof(int) ); 
        pCrr +=  auxInfoVect.size()*sizeof(int);


        // decompression statistics
        int cntr_orig = 0;
        int cntr_comprs = 0;
        int cntr_runLengths = 0;

        if(r_signum == 0) {  // strting with "no"-sequence
            cntr_orig += auxInfoVect[cntr_runLengths];
            for(int i = 0; i < cntr_orig; i++) {
                output[i] = 0;
            }
            cntr_runLengths++;
        }
        // now alternate (i) "yes"-items (ii) "no" items
        while (cntr_orig < r_original_size) {
            // "yes"-items
            for(int i = cntr_orig; i < cntr_orig + auxInfoVect[cntr_runLengths]; i++) {
                //if(cntr_comprs == _shrinkedSize) throw;
                output[i] = pShrinkedVector[cntr_comprs];
                cntr_comprs++;
            }
            cntr_orig += auxInfoVect[cntr_runLengths];
            cntr_runLengths++;

            if(cntr_orig == r_original_size) break;
            // "no"-items
            for(int i = cntr_orig; i < cntr_orig + auxInfoVect[cntr_runLengths]; i++) {
                output[i] = 0;       
            }
            cntr_orig += auxInfoVect[cntr_runLengths];
            cntr_runLengths++;
        }
        return r_original_size;
    } // decompression_strategy


    virtual void add_strategy(VarTYPE* output, int output_size, const void* sourceBuffer) {

        const void* pCrr = sourceBuffer;
        const VarTYPE* pShrinkedVector;

        // received data statistics
        std::vector<int> auxInfoVect;
        int r_original_size=0;
        int r_compressed_size=0;
        int r_auxInfoVect_size=0;
        int r_signum=0;

        // originalVectorSize , int
        r_original_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        if(r_original_size>output_size){
            throw std::runtime_error("Decompression vector size is bigger than output buffer!");
        }
        
        // shrinkedVectorSize , int
        r_compressed_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);

        // _shrinkedSize VarTYPE, decompress later from this position
        pShrinkedVector = reinterpret_cast<const VarTYPE *>(pCrr);
        pCrr += r_compressed_size*sizeof(VarTYPE);

        //  run-Length-vector Size, int
        r_auxInfoVect_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        auxInfoVect.resize(r_auxInfoVect_size);
    
        // signumFlag, int
        r_signum = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        
        // _runLengthsSize integers, the contents of _auxInfoVect[]-vector
        std::memcpy(auxInfoVect.data(), pCrr, auxInfoVect.size()*sizeof(int) ); 
        pCrr +=  auxInfoVect.size()*sizeof(int);


        // decompression statistics
        int cntr_orig = 0;
        int cntr_comprs = 0;
        int cntr_runLengths = 0;

        if(r_signum == 0) {  // strting with "no"-sequence
            cntr_orig += auxInfoVect[cntr_runLengths];
            cntr_runLengths++;
        }
        // now alternate (i) "yes"-items (ii) "no" items
        while (cntr_orig < r_original_size) {
            // "yes"-items
            for(int i = cntr_orig; i < cntr_orig + auxInfoVect[cntr_runLengths]; i++) {
                //if(cntr_comprs == _shrinkedSize) throw;
                output[i] += pShrinkedVector[cntr_comprs];
                cntr_comprs++;
            }
            cntr_orig += auxInfoVect[cntr_runLengths];
            cntr_runLengths++;

            if(cntr_orig == r_original_size) break;
            // "no"-items
            cntr_orig += auxInfoVect[cntr_runLengths];
            cntr_runLengths++;
        }
    } 

};

/***************************************
 * CompressorIndexPairs
 * Apply index-value pairs compression
 ***************************************/
template <class VarTYPE>
class CompressorIndexPairs : public Compressor<VarTYPE> {
public:
    CompressorIndexPairs<VarTYPE>() 
        : _topK(0)
        {}

    CompressorIndexPairs<VarTYPE>(float topK) 
        : _topK(topK)
        {}

    virtual Compressor<VarTYPE>* clone() const {
        return new CompressorIndexPairs<VarTYPE>(*this);
    }

protected:
    float _topK;

    virtual int compression_strategy(void* output, const VarTYPE* data_vector, VarTYPE* update_vector, const int vector_size, VarTYPE threshold) {
        int compressed_size = 0;

        void* pCrr = output;
        // write uncompressed vector size
        *(reinterpret_cast<int *>(pCrr)) = vector_size;
        pCrr += sizeof(int);
        // write compressed vector size later
        int* pCompressed_size = reinterpret_cast<int *>(pCrr);
        pCrr += sizeof(int);
        
        // write data
        if(data_vector != NULL && update_vector != NULL){
            for(int i=0; i<vector_size; ++i){
                update_vector[i] += data_vector[i];
                if( std::abs(update_vector[i]) > threshold ) {
                    *(reinterpret_cast<int *>(pCrr)) = i;
                    pCrr += sizeof(int);
                    *(reinterpret_cast<VarTYPE *>(pCrr)) = update_vector[i];
                    pCrr += sizeof(VarTYPE);
                    ++compressed_size;
                    // erase written element from buffer
                    update_vector[i] = 0;
                }
            }
        }
        else if(update_vector != NULL){
            for(int i=0; i<vector_size; ++i){
                if( std::abs(update_vector[i]) > threshold ) {
                    *(reinterpret_cast<int *>(pCrr)) = i;
                    pCrr += sizeof(int);
                    *(reinterpret_cast<VarTYPE *>(pCrr)) = update_vector[i];
                    pCrr += sizeof(VarTYPE);
                    ++compressed_size;
                    // erase written element from buffer
                    update_vector[i] = 0;
                }
            }
        }
        else if(data_vector != NULL){
            for(int i=0; i<vector_size; ++i){
                if( std::abs(data_vector[i]) > threshold ) {
                    *(reinterpret_cast<int *>(pCrr)) = i;
                    pCrr += sizeof(int);
                    *(reinterpret_cast<VarTYPE *>(pCrr)) = data_vector[i];
                    pCrr += sizeof(VarTYPE);
                    ++compressed_size;
                }
            }
        }
        *pCompressed_size = compressed_size;

        //gaspi_printf("compression: %f\n", 1.0 - (float)compressed_size/(float)vector_size);

        return (char const volatile*)pCrr-(char const volatile*)output;
    }

    virtual int decompression_strategy(VarTYPE* output, int output_size, const void* input) {
        // zero output
        std::memset(output, 0, output_size*sizeof(VarTYPE));
        const void* pCrr = input;
        // uncompressed vector size
        int r_uncompressed_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        if(r_uncompressed_size>output_size){
            char msg[100];
            sprintf(msg, "Decompression vector size (%d) is bigger than output buffer (%d)!", r_uncompressed_size, output_size );
            throw std::runtime_error(msg);
        }
        // load compressed vector size value
        int r_compressed_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        // load data into vector
        for(int i=0; i<r_compressed_size; ++i){
            int index = *(reinterpret_cast<const int *>(pCrr));
            pCrr += sizeof(int);
            output[index] = *(reinterpret_cast<const VarTYPE *>(pCrr));
            pCrr += sizeof(VarTYPE);
        }
        int sizeBytes = (char const volatile*)pCrr-(char const volatile*)input;
        return sizeBytes;
        //return r_uncompressed_size;
    }

    virtual void add_strategy(VarTYPE* output, int output_size, const void* sourceBuffer) {
        const void* pCrr = sourceBuffer;
        // uncompressed vector size
        int r_uncompressed_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        if(r_uncompressed_size>output_size){
            throw std::runtime_error("Decompression vector size is bigger than output buffer!");
        }
        // load compressed vector size value
        int r_compressed_size = *(reinterpret_cast<const int *>(pCrr));
        pCrr += sizeof(int);
        // load data into vector
        for(int i=0; i<r_compressed_size; ++i){
            int index = *(reinterpret_cast<const int *>(pCrr));
            pCrr += sizeof(int);
            output[index] += *(reinterpret_cast<const VarTYPE *>(pCrr));
            pCrr += sizeof(VarTYPE);
        }
    }


}; //class Compressor

#endif //COMPRESSOR_H
