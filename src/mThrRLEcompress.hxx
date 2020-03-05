// mThrRLEcompress.hxx : multi-threaded RLE comression, definitions

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring> // memcpy
#include <memory>

#ifndef M_THREADED_RLE_COMPRESSION_HXX
#define M_THREADED_RLE_COMPRESSION_HXX

namespace compressed_exchange {

  template <class VarTYPE>
  class MultiThreadedRLE {

    private:

       // size of and a reference to the original vector
       int _origSize;
       const VarTYPE* _origVector;

       // treshold value used to compress the vector 
       VarTYPE _treshold;

       // flag indicating the "sign" ("yes" or "no") of the starting
       // run-length-sequence. The signs then alternatingly change.
       // "yes"-sequence <-> signumFlag = 1
       // "no"-sequence <-> signumFlag = 0       
       // prefix "glb" -> signum flag for the global RLE-sequence
       int & _glb_signumFlag;

       // The compressed vector, with items/values greater than treshold,   
       // i.e. _shrinkedVector[i] >= treshold, i = 0..(_shrinkedSize-1).
       // This is the vector to be transferred, together with the
       // auxiliary-info-vector, e.g. of run-lengthg codes, see below 
       std::vector<VarTYPE> & _glb_shrinkedVect; // global compressed vector

       // auxiliary  info vector, i.e. containing auxiliary information, 
       // in addition to the compressed vector values. This means
       // - vector of run-lenth codes, for run-length-encoding
       // - vector containing the  indices of the non-zeros, in sparse-indexing 
       std::vector<int>  &  _glb_runLengthsVect; // global RLE-sequence 

       // number of threads, pin_pattern
       int _numThreads;
       const int* _pinPattern;

       //  thread-local structures
       //
       // start index in the original-vector for each thread
       std::unique_ptr< int [] > _startIdx_thr; 

       // array of signum-flags,  one item per each thread
       std::unique_ptr< int [] >  _signumFlag_thr;

       // array of sizes of the run-lengths-vectors, one item per each thread
       std::unique_ptr< int [] >  _runLengthsVectSize_thr;

       // array of sizes of the shrinked-vectors, one item per each thread
       std::unique_ptr< int [] >  _shrinkedVectSize_thr;

       // array of run-lengths-vectors, one vector per each thread
       std::unique_ptr< std::vector<int> [] > _runLengthsVect_thr;

       // array of shrinked-vectors, one vector per each thread
       std::unique_ptr< std::vector<VarTYPE> [] > _shrinkedVect_thr;
       
       //const int* _pinPattern; // [_numThreads], pin pattern   
       //void pinnThisThread(int gpiRank, int threadID, int coreId);
       //void setThreadParameters_compressVectorVhunk(gaspi_rank_t rank, int threadID);

       //static void* threadRoutine_compressVectorChunk(void *arg);

       void allocateThreadRelatedArrays();
       void partitionTheOriginalVector();

    public:

      MultiThreadedRLE(
		     int origSize
		   , const VarTYPE * origVector
		   , VarTYPE  treshold
		   , int & signumFlag
		   , std::vector<VarTYPE> & glbShrinkedVect
		   , std::vector<int>  & glbRunLengthsVect
		   , int nThreads
		     );

     ~MultiThreadedRLE();

     void setPinPatternForTheThreads(
	    int nThreads,       // number of threads per rank
            const int * pinPattern); // pin pattern


    void compress( int & shrinkedVectSize, 
                   int & runLengthsVectSize);
      

  }; // class MultiThreadedRLE
 
} // namespace compressed_exchange  

//#include <mThrRLEcompress.cxx>

#endif // M_THREADED_RLE_COMPRESSION_HXX 


