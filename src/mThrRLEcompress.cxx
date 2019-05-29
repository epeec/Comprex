// mThrRLEcompress.cxx : multi-threaded RLE compression, implementation

#include <mThrRLEcompress.hxx>


namespace compressed_exchange {

  template <class VarTYPE>
  MultiThreadedRLE<VarTYPE>::MultiThreadedRLE(
		     int origSize
		   , const VarTYPE * origVector
		   , VarTYPE  treshold
		   , int & signumFlag
		   , std::vector<VarTYPE> & glbShrinkedVect
		   , std::vector<int>  & glbRunLengthsVect
		   , int nThreads)
    :
      _origSize(origSize)
    , _origVector(origVector)
    , _treshold(treshold)
    , _glb_signumFlag(signumFlag) 
    , _glb_shrinkedVect(glbShrinkedVect)
    , _glb_runLengthsVect(glbRunLengthsVect)  
    , _numThreads(nThreads)
      , _startIdx_thr()
      , _signumFlag_thr()
      , _runLengthsVectSize_thr()
      , _shrinkedVectSize_thr()
      , _runLengthsVect_thr()
      , _shrinkedVect_thr()
  {

     allocateThreadRelatedArrays();
     partitionTheOriginalVector();
  }

  
  template <class VarTYPE>
  MultiThreadedRLE<VarTYPE>::~MultiThreadedRLE()
  {
  }
  
  template <class VarTYPE>
  void
  MultiThreadedRLE<VarTYPE>::allocateThreadRelatedArrays()
  {
    _startIdx_thr = std::unique_ptr<int []> (new int [_numThreads+1]);
    _signumFlag_thr = std::unique_ptr<int []> (new int [_numThreads]);
    _runLengthsVectSize_thr = std::unique_ptr<int []> (new int [_numThreads]);
    _shrinkedVectSize_thr = std::unique_ptr<int []> (new int [_numThreads]);
    _runLengthsVect_thr = std::unique_ptr< std::vector<int> []> 
                                      (new std::vector<int> [_numThreads]);
    _shrinkedVect_thr = std::unique_ptr< std::vector<VarTYPE> []> 
                                      (new std::vector<VarTYPE> [_numThreads]);

  }

  template <class VarTYPE>
  void
  MultiThreadedRLE<VarTYPE>::partitionTheOriginalVector()
  {

    int size;
    _startIdx_thr[0] = 0;
    for(int i = 1; i < _numThreads; i++) {
         size = _origSize/_numThreads;
         if(i < _origSize % _numThreads ) size++;
         _startIdx_thr[i] = _startIdx_thr[i-1]+ size;
    }
    _startIdx_thr[_numThreads] = _origSize;

  }  

  template <class VarTYPE>
  void
  MultiThreadedRLE<VarTYPE>::compress( int & shrinkedVectSize, 
                                       int & runLengthsVectSize)
  {

       
     shrinkedVectSize = _glb_shrinkedVect.size();
     runLengthsVectSize = _glb_runLengthsVect.size();
  }


} // namespace compressed_exchange
