#ifndef COMPRESSEDVECTOR_HXX
#define COMPRESSEDVECTOR_HXX

#include <vector>
#include <cstring>
#include <string>
#include <iostream>

/***************************************
 * CompressedVector
 * Base Class for Compressed Vectors
 * Contains Comprex (GPI) specific information
 ***************************************/
template <class VarTYPE>
class CompressedVector{
public:
    //CompressedVector() {}
    virtual void writeBuffer(void* buffer) const =0; // create a blob in buffer from the Compressed Vector
    virtual void loadBuffer(void* buffer)=0; // load data from a blob located in buffer
    virtual int calcSizeBytes() const =0; // calculates the size of the transfered buffer in bytes
    virtual int uncompressed_size() const =0; // get the number of elements in the uncompressed vector
protected:
}; //class CompressedVector


/***************************************
 * CompressedVectorNone
 * Uncompressed Vector
 * Contains Comprex (GPI) specific information
 ***************************************/
template <class VarTYPE>
class CompressedVectorNone : public CompressedVector<VarTYPE>{
public:
    // uncompressed vector
    std::vector<VarTYPE> vect;

    CompressedVectorNone(){
    }

    CompressedVectorNone(const std::vector<VarTYPE>* vect)
            : vect(*vect){
    }

    virtual void writeBuffer(void* buffer) const {
        void* pCrr = buffer;
        // write size of vector
        *(reinterpret_cast<int *>(pCrr)) = vect.size();
        pCrr += sizeof(int);
        // write data
        std::memcpy( pCrr, vect.data(), vect.size()*sizeof(VarTYPE) ); 
        pCrr +=  vect.size()*sizeof(VarTYPE);
    }

    virtual void loadBuffer(void* buffer) {
        void* pCrr = buffer;
        // load and adjust size of vector
        int r_size = *(reinterpret_cast<int *>(pCrr));
        pCrr += sizeof(int);
        vect.resize(r_size);
        // load data into vector
        std::memcpy(vect.data(), pCrr, vect.size()*sizeof(VarTYPE) ); 
        pCrr +=  vect.size()*sizeof(VarTYPE);
    }

    virtual int calcSizeBytes() const {
        int sizeBytes=0;
        sizeBytes += sizeof(int); // vect size
        sizeBytes += vect.size()*sizeof(VarTYPE); // vect data
        return sizeBytes;
    }

    // Getter
    std::vector<VarTYPE> getVect() const { return vect; }

    int uncompressed_size() const {return vect.size(); }

protected:
}; //class CompressedVector


/***************************************
 * CompressedVectorRLE
 * Abstract Base Class for Compressed Vectors
 ***************************************/
template <class VarTYPE>
class CompressedVectorRLE : public CompressedVector<VarTYPE>{
public:
    CompressedVectorRLE() 
            : original_size(0) {
    }

    CompressedVectorRLE(int original_size, int signum, const std::vector<int>& auxInfoVect, const std::vector<VarTYPE>& compressedVect) 
            : original_size(original_size), signum(signum), auxInfoVect(auxInfoVect), compressedVect(compressedVect) {
    }

    virtual void writeBuffer(void* buffer) const {
        //void* pBuffBegin = srcBuff.address();
        //void* pCrr =   pBuffBegin;
        void* pCrr = buffer;

        // originalVectorSize , int
        *(reinterpret_cast<int *>(pCrr)) = original_size;
        pCrr += sizeof(int);

        // shrinkedVectorSize , int
        *(reinterpret_cast<int *>(pCrr)) = compressedVect.size();
        pCrr += sizeof(int);

        //  run-Length-vector Size, int
        *(reinterpret_cast<int *>(pCrr)) = auxInfoVect.size();
        pCrr += sizeof(int);
    
        // signumFlag, int
        *(reinterpret_cast<int *>(pCrr)) = signum;
        pCrr += sizeof(int);

        // _runLengthsSize integers, the contents of _auxInfoVect[]-vector
        std::memcpy( pCrr, auxInfoVect.data(), auxInfoVect.size()*sizeof(int) ); 
        pCrr +=  auxInfoVect.size()*sizeof(int);

        // _shrinkedSize VarTYPE, the contents of _shrinkedVect[]-vector
        std::memcpy(pCrr, compressedVect.data(), compressedVect.size()*sizeof(VarTYPE) ); 
        pCrr +=  compressedVect.size()*sizeof(VarTYPE);
    }

    virtual void loadBuffer(void* buffer) {
        //void* pBuffBegin = targBuff.address();
        //void* pCrr =   pBuffBegin;
        void* pCrr = buffer;

        // received data statistics
        int r_original_size=0;
        int r_compressed_size=0;
        int r_auxInfoVect_size=0;
        int r_signum=0;

        // originalVectorSize , int
        r_original_size = *(reinterpret_cast<int *>(pCrr));
        pCrr += sizeof(int);
        original_size=r_original_size;
        /*
        if(r_original_size != original_size) {
            //printf("\n [%d] sendr-rank-orig-size:%d my-orig_size:%d \n",
            //        ComprEx<VarTYPE>::_gpiCxx_context.rank().get(), 
            //        *(reinterpret_cast<int *>(pCrr)), ComprEx<VarTYPE>::_origSize);
            throw std::runtime_error ("sender-rank uncompressed vector size differs from the receiver one.");
        }*/
        
        // shrinkedVectorSize , int
        r_compressed_size = *(reinterpret_cast<int *>(pCrr));
        pCrr += sizeof(int);
        compressedVect.resize(r_compressed_size);

        //  run-Length-vector Size, int
        r_auxInfoVect_size = *(reinterpret_cast<int *>(pCrr));
        pCrr += sizeof(int);
        auxInfoVect.resize(r_auxInfoVect_size);
    
        // signumFlag, int
        r_signum = *(reinterpret_cast<int *>(pCrr));
        pCrr += sizeof(int);
        signum = r_signum;
        
        // _runLengthsSize integers, the contents of _auxInfoVect[]-vector
        std::memcpy(auxInfoVect.data(), pCrr, auxInfoVect.size()*sizeof(int) ); 
        pCrr +=  auxInfoVect.size()*sizeof(int);

        // _shrinkedSize VarTYPE, the contents of _shrinkedVect[]-vector
        std::memcpy(compressedVect.data(), pCrr, compressedVect.size()*sizeof(VarTYPE) ); 
        pCrr += compressedVect.size()*sizeof(VarTYPE);
    }

    int calcSizeBytes() const{
        int buffSizeBytes = 0;
        // start with 4 integers:
        //    int original Vector Size
        //    int shrinkedVectorSize
        //    int runLengthSize
        //    int signumFlag
        buffSizeBytes += 4*sizeof(int);
        // then the runLengths[] array, i.e. #_auxInfoVectSize integers
        buffSizeBytes += auxInfoVect.size() * sizeof(int);
        // then the compressed vector values, i.e. #_shrinkedSize VarTYPE-s
        buffSizeBytes += compressedVect.size() * sizeof(VarTYPE);
        return buffSizeBytes;
    }

    // Getters
    int get_signum() const { return signum;}

    int uncompressed_size() const {return original_size;}
    int get_original_size() const {return uncompressed_size();}

    const std::vector<int>* get_auxInfoVect_p() const {return &auxInfoVect;}

    int get_auxInfoVect(int i) const {return auxInfoVect[i];}

    const std::vector<VarTYPE>* get_compressedVect_p() const {return &compressedVect;}

    VarTYPE get_compressedVect(int i) const {return compressedVect[i];}

    int get_compressedVect_size() const { return compressedVect.size();}

    // sprint vector
    std::string sprint() const {
        std::string output;

        // RLE auxilliary
        // start with the signum flag
        output += "RLE aux. Vector\n";
        
        output += "signumFlag: "+std::to_string(signum)+"\n";
        // size of the run-lengths sequence 
        output += "run-lengths-sequence size: " + std::to_string(auxInfoVect.size()) +"\n";
        // now the run-length sequence
        output += " runLength[ ";
        for ( auto it = auxInfoVect.begin(); it != auxInfoVect.end(); ++it) {
            output += std::to_string(*it) + ", ";
        } 
        output.pop_back(); output.pop_back();
        output+="]\n";
        output+="\n";

        // RLE numbers
        output += "RLE compressed Vector\n";
        output += "size: " + std::to_string(compressedVect.size())+"\n";
        // now the run-length sequence
        output += "compressed[ ";
        for ( auto it = compressedVect.begin(); it != compressedVect.end(); ++it) {
            output += std::to_string(*it) + ", ";
        } 
        output.pop_back(); output.pop_back();
        output += "]\n";
        output+="\n";

        // Additional Information
        output += "Additional Information\n";
        output += "GPI Buffer Size: "+std::to_string(calcSizeBytes())+" Bytes\n";
        
        return output;
    }
    
protected:
    int original_size; // safety feature, checks if decompression is successful
    int signum;
    // auxiliary  info vector, i.e. containing auxiliary information, 
    // in addition to the compressed vector values. This means
    // - vector of run-lenth codes, for run-length-encoding
    // - vector containing the  indices of the non-zeros, in sparse-indexing 
    // TODO: this is part of a compression strategy, not comprex
    std::vector<int> auxInfoVect; 
    // The compressed vector, with items/values greater than treshold,   
    // i.e. _shrinkedVector[i] >= treshold, i = 0..(_shrinkedSize-1).
    // This is the vector to be transferred, together with the
    // auxiliary-info-vector, e.g. of run-lengthg codes, see below 
    std::vector<VarTYPE> compressedVect;
    
}; //class CompressedVector



#endif //COMPRESSEDVECTOR_HXX