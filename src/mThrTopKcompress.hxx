// mThrTopKcompress.hxx : multi-threaded Top-K-percents comression, definitions
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring> // memcpy
#include <memory>
#include <cstdlib>

#include <algorithm> // std::sort

#ifndef M_THREADED_TOP_K_COMPRESSION_HXX
#define M_THREADED_TOP_K_COMPRESSION_HXX

namespace compressed_exchange {


  template <class VarTYPE> 
  struct PairIndexValue {
      int idx;
      VarTYPE val;
  };


   // forward declaration
   template  <class VarTYPE> 
   class  MultiThreadedTopK; // forward declaration
  
  template <class VarTYPE> 
  struct  ThreadParmeters_compress
   {

     int threadIdx;
     MultiThreadedTopK<VarTYPE> *pClassMThrTopK;
   }; //ThreadParmeters_compress

  template <class VarTYPE> 
  struct  ThreadParmeters_uncompress
   {

     int threadIdx;

     MultiThreadedTopK<VarTYPE> *pClassMThrTopK;

     //..    
   }; //ThreadParmeters_uncompress


  template <class VarTYPE>
  class MultiThreadedTopK {

    private:

       // GASPI- (or GaspiCxx-) rank
       // (One needs the rank only for debug)
       int _rank;

       // size of and a reference to the original vector and the rests vector
       int _origSize;

       const VarTYPE* _origVector;
       VarTYPE* _restsVector;

       // shrinked size, i.e. overall size of the cpmressed vector.
       // It must have been already calculated in ComprExTopK-class
       int _shrinkedSize;

       // Reference to the global (shrinked) vector of pairs <idx, value>
       // This is the "output" vector, result of the m-threaded compression
       std::vector<PairIndexValue<VarTYPE> > & _vecPairsGlb;
       //PairIndexValue<VarTYPE> * _pVectPairsGlb;

       // number of threads, pin_pattern
       // If no pin-pattern has been provided by the "user"
       // then leave the pinning to the OS
       int _numThreads;
       pthread_t *_thread; // array of size [_numThreads]
       const int* _pinPattern;  // pin pattern for the threads

       //  thread-local structures
       //
       // start index in the original-vector for each thread
       std::unique_ptr< int [] > _startIdx_thr; 

       // start index in the original-vector for each thread
       std::unique_ptr< int [] > _startIdx_comprsdV_thr; 

       // working vectors for each thread
       std::unique_ptr< 
          std::vector<PairIndexValue<VarTYPE> > []>
                                               _vectPairs_thr;



       // compress
       std::unique_ptr< ThreadParmeters_compress<VarTYPE> [] >
                                            _pThreadParams_compress;
 
       // uncompress
       const PairIndexValue<VarTYPE > *_pPairsVect;

       std::unique_ptr< ThreadParmeters_uncompress<VarTYPE> [] >
                                            _pThreadParams_uncompress;


       VarTYPE* _destVector;

       void allocateThreadRelatedArrays();
       void partitionTheOriginalVector();
       void partitionTheCompressedVector(int vectSize); 

       void pinThisThread(int threadIdx, int coreID) const;

       void setThreadParameters_compress(int threadIdx);
       void setThreadParameters_uncompress(int threadIdx);
      

       static bool sortPredicate_absValue(const PairIndexValue<VarTYPE> & obj1, 
			                  const PairIndexValue<VarTYPE> & obj2)
       {
	 return std::abs(obj1.val) > std::abs(obj2.val);
       }


    public:

      // constructor used for the compression-phase
      // 
      // Note: vecPairs is a reference to ComprExTopK::_vectPairs
      // Threrefore:
      //  -> call this constructor only after ComprExTopK::_vectPairs
      //     has been instantiated (and even resized), see 
      //     ComprExTopK<VarTYPE>::compress_and_p2pVectorWriteRemote()
      //
      //  Note that this class modifies  ComprExTopK::_vectPairs,
      //  thus, we need a reference to it and this reference should be
      //  instantiated here, in the constructor
      //  Note also that a pointer here (calling std::vector::data() on
      //   ComprExTopK::_vectPairs) will not help, we need a reference.
      MultiThreadedTopK(
		    int rank
		   ,  int origSize
		   , const VarTYPE * origVector
		   , VarTYPE * restsVector
		   , int shrinkedSize
		   , std::vector<PairIndexValue<VarTYPE> > & vecPairs
		   , int nThreads
                   , pthread_t * pThreads
		   , const int* pinPattern = NULL
		   );

    
      // constructor used for the expansion-phase  
      // Note: vecPairs is a reference to ComprExTopK::_vectPairs
      // Threrefore:
      //  -> call this constructor only after ComprExTopK::_vectPairs
      //     has been instantiated (and even resized) in the 
      //     ComprExTopK-routines
      MultiThreadedTopK(
		    int rank
		   ,  int origSize  
		   , int shrinkedSize
		   , std::vector<PairIndexValue<VarTYPE> > & vecPairs
		   , int nThreads
                   , pthread_t * pThreads
		   , const int* pinPattern = NULL
		   );


     ~MultiThreadedTopK();


    void classThreadRoutine_compress(int threadID);
    void classThreadRoutine_uncompress(int threadID);

    void compress( );

    void uncompress( std::unique_ptr<VarTYPE []>  & vector );
      

  }; // class MultiThreadedTopK

  
  template <class VarTYPE>
  static void *thread_func_compressTopK(void *args) {
    
    struct ThreadParmeters_compress<VarTYPE> *pPars =
      static_cast<struct ThreadParmeters_compress<VarTYPE> *> (args);
            
    int threadID = pPars->threadIdx;
    MultiThreadedTopK<VarTYPE> *pClassInstance =   pPars->pClassMThrTopK;

    //printf("\n tI:%d in the outer-static-thread roitine", threadID);
    pClassInstance->classThreadRoutine_compress(threadID);
    
  }
  
  template <class VarTYPE>
  static void *thread_func_uncompressTopK(void *args) {

    struct ThreadParmeters_uncompress<VarTYPE> *pPars =
    static_cast<struct ThreadParmeters_uncompress<VarTYPE> *> (args);
            
    int threadID = pPars->threadIdx;
    MultiThreadedTopK<VarTYPE> *pClassInstance =   pPars->pClassMThrTopK;

    pClassInstance->classThreadRoutine_uncompress(threadID);

  }
 
} // namespace compressed_exchange  

#endif // M_THREADED_TOP_K_COMPRESSION_HXX 

