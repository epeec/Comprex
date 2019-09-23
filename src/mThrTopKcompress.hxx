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

     //..    
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

       ThreadParmeters_uncompress<VarTYPE> *_pThreadParams_uncompress;

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

     ~MultiThreadedTopK();


    void classThreadRoutine_compress(int threadID);
    void classThreadRoutine_uncompress(int threadID);

    void compress( 
    		std::vector< PairIndexValue <VarTYPE> > & glbShrinkedVect);

    //void compress( int rank, int nThreads);

    void uncompress(
               int shrinkedSize
	     , std::vector< PairIndexValue <VarTYPE> > & glbShrinkedVect
	     , std::unique_ptr<VarTYPE []>  & vector );
      

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


  }
 
} // namespace compressed_exchange  

#endif // M_THREADED_TOP_K_COMPRESSION_HXX 

