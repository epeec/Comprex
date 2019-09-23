// mThrTopKcompress.cxx : multi-threaded Top-K-percents comression, implemention
#include <mThrTopKcompress.hxx>


namespace compressed_exchange {

  template <class VarTYPE>
  MultiThreadedTopK<VarTYPE>::MultiThreadedTopK(
                   int rank
		   , int origSize
		   , const VarTYPE * origVector
                   , VarTYPE * restsVector
                   , int shrinkedSize
                   , std::vector<PairIndexValue<VarTYPE> > & vecPairs
		   , int nThreads
                   , pthread_t * pThreads
                   , const int* pinPattern
                   )
    :
      _rank(rank)
    , _origSize(origSize)
    , _origVector(origVector)
    , _restsVector(restsVector)
    , _shrinkedSize(shrinkedSize) 
    , _vecPairsGlb(vecPairs)
    , _numThreads(nThreads)
    , _thread(pThreads)
    , _pinPattern(pinPattern)
    , _startIdx_thr()
    , _startIdx_comprsdV_thr()
  {

     allocateThreadRelatedArrays();
  
     partitionTheOriginalVector();

     partitionTheCompressedVector(_shrinkedSize);

//printf("\n [%d] MultiThreadedTopK-constructor, start %d threads \n",
//       _rank, _numThreads);
//for(int i = 0; i < 10; i++) printf("\n _restsVec[%d]=%d  origVect[%d]=%d", 
//         i, _restsVector[i], i, _origVector[i]);
  }

  
  template <class VarTYPE>
  MultiThreadedTopK<VarTYPE>::~MultiThreadedTopK()
  {
  }
  
  template <class VarTYPE>
  void
  MultiThreadedTopK<VarTYPE>::allocateThreadRelatedArrays()
  {

 
    _pThreadParams_compress = 
        std::unique_ptr<ThreadParmeters_compress<VarTYPE> []> 
           (new ThreadParmeters_compress<VarTYPE> [_numThreads]);

    
    _startIdx_thr = std::unique_ptr<int []> (new int [_numThreads+1]);

    _startIdx_comprsdV_thr = std::unique_ptr<int []> (new int [_numThreads+1]);

    _vectPairs_thr =  
        std::unique_ptr< std::vector< PairIndexValue<VarTYPE> > [] >
         (new std::vector< PairIndexValue<VarTYPE> > [_numThreads]);

  }


  template <class VarTYPE>
  void
  MultiThreadedTopK<VarTYPE>::partitionTheOriginalVector()
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
  MultiThreadedTopK<VarTYPE>::partitionTheCompressedVector(int vectSize)
  {
    
    int size;
    _startIdx_comprsdV_thr[0] = 0;
    for(int i = 1; i < _numThreads; i++) {
         size = vectSize/_numThreads;
         if(i < vectSize % _numThreads ) size++;
         _startIdx_comprsdV_thr[i] = _startIdx_comprsdV_thr[i-1]+ size;
    }
    _startIdx_comprsdV_thr[_numThreads] = vectSize;

  }  

  template <class VarTYPE>
  void
  MultiThreadedTopK<VarTYPE>::pinThisThread(int threadIdx, int coreID) const
  {
      // pin the threads, one-to-one (i.e., thread-to-core == 1-to-1)
      cpu_set_t cpuset;
      pthread_t thread;

      thread = pthread_self();

      CPU_ZERO(&cpuset);
      CPU_SET(coreID, &cpuset);

      int ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
      if (ret != 0) {
	std::cout  << " rank:" << _rank << " thread[" << threadIdx 
		  << "], ERROR setting affinity mask !!!, errNo:" 
                  << ret << std::endl;
      }

      ret = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
      if (ret != 0) {
        std::cout  << " rank:" << _rank << " thread[" << threadIdx
		  << "], ERROR getting affinity mask !!!, errNo:" 
                  << ret << std::endl;
      }
      
      if (CPU_ISSET(coreID, &cpuset)) 
         printf(" [%d]  coreID:%d in cpuset, associated with thread: %d\n", 
		_rank, coreID, threadIdx);
  }




  template <class VarTYPE>
  void 
  MultiThreadedTopK<VarTYPE>::setThreadParameters_compress(int threadIdx)
  {

     _pThreadParams_compress[threadIdx].threadIdx = threadIdx;
     _pThreadParams_compress[threadIdx].pClassMThrTopK = this;
  }


  template <class VarTYPE>
  void 
  MultiThreadedTopK<VarTYPE>::classThreadRoutine_compress(int threadID)
  {
     // pin the thread [ if requested (set a global flag ?!) }
    if(_pinPattern != NULL ) {
      //printf("\n [%d] pint thread:%d to coreID:%d, start compression \n", 
      //  _rank, threadID,  _pinPattern[threadID]);
      pinThisThread(threadID, _pinPattern[threadID] );
    }  

    // resize the thread-own work-vector _vectPairs_thr[threadID]
    int workSize_thisThread = _startIdx_thr[threadID+1]
                                      - _startIdx_thr[threadID];
    (_vectPairs_thr[threadID]).resize(workSize_thisThread);

    /*
printf("\n [%d] thread:%d this-thread-> wrkSize_vactPairs:%d  start_in_idx:%d start_outp_idx:%d \n", 
       _rank, threadID, workSize_thisThread, 
     _startIdx_thr[threadID],  _startIdx_comprsdV_thr[threadID] );


if(threadID == 0) {
 printf("\n----------------\n");
for(int i = 0; i < 10; i++) printf("\n thr:%d _restsVec[%d]=%d  origVect[%d]=%d", 
	 threadID, i, _restsVector[i], i,  _origVector[i]);
 printf("\n----------------\n");
}//if(threadID == 0) {
    */

    // fill-in the data for this thread and and sort
    for(int i = 0; i < workSize_thisThread; i++) {
      int glbIdx = _startIdx_thr[threadID] + i; 
      _restsVector[glbIdx] += _origVector[glbIdx];
      (_vectPairs_thr[threadID])[i].idx = glbIdx;
      (_vectPairs_thr[threadID])[i].val = _restsVector[glbIdx];
    }
 
//printf("\n [%d] thread:%d after setting  this-thread-vectPairs, sort now \n", 
//	   _rank, threadID);

    // now sort the items in _vectPairs_thr[threadID]   
    std::sort((_vectPairs_thr[threadID]).begin(), 
	   (_vectPairs_thr[threadID]).end(),  sortPredicate_absValue);


    /*    
typedef typename std::vector<PairIndexValue<VarTYPE> >::const_iterator 
                                                        MyIterType;
int cntr = 0;    
for ( MyIterType it = (_vectPairs_thr[threadID]).begin(); 
       it != (_vectPairs_thr[threadID]).end(); ++it) {
   //std::out << " sortedVect[" << cntr << "], ->origIdx: " 
   //         << it->idx << " , ->orig_val = " 
   //         << it->val  //*it 
   //         << std::endl;

  printf("\n tI:%d sorted own-vactPairs[%d]  ->glbIdx:%d ->(orig_)val:%d",
	 threadID, cntr, it->idx, it->val);
       cntr++;  
}// for ( MyIterType it = 
    */

    // copy the first (#nOutItems) number of items to the output vector
    int nOutItems =  _startIdx_comprsdV_thr[threadID+1] 
                                - _startIdx_comprsdV_thr[threadID]; 

//printf("\n [%d] thread:%d this-thread-nOutItems:%d, write in the out vect (sz:%d)\n", 
//       _rank, threadID, nOutItems, _shrinkedSize);

    for(int i = 0; i < nOutItems; i++) {
      int glbIdx = _startIdx_comprsdV_thr[threadID]+i;

      /*
printf("\n [%d] thr:%d lclI:%d glbI:%d vPairsGlb[%d].idx=%d VPairsGlb[%d].val=%d  \n", 
       _rank, threadID, i, glbIdx, glbIdx, 
       (_vectPairs_thr[threadID])[i].idx, glbIdx, 
       (_vectPairs_thr[threadID])[i].val);
      */

      _vecPairsGlb[glbIdx].idx = (_vectPairs_thr[threadID])[i].idx;
      _vecPairsGlb[glbIdx].val = (_vectPairs_thr[threadID])[i].val;
    } 
 
  }
    

  template <class VarTYPE>
  void 
  MultiThreadedTopK<VarTYPE>::classThreadRoutine_uncompress(int threadID)
  {


  }


  template <class VarTYPE>
  void 
  MultiThreadedTopK<VarTYPE>::setThreadParameters_uncompress(int threadIdx)
  {

  }



  template <class VarTYPE>
  void
  MultiThreadedTopK<VarTYPE>::compress(
  	        std::vector< PairIndexValue <VarTYPE> > & glbShrinkedVect )
  {
     for(int i=0;  i < _numThreads; i++)  setThreadParameters_compress(i);
     
     int stat;
     for(int i = 0; i < _numThreads; i++) {
       stat = pthread_create( &(_thread[i]), NULL, 
              (thread_func_compressTopK<VarTYPE>), 
	      static_cast<void *> ( &(_pThreadParams_compress[i])) );
       
       if(stat != 0) printf(" err creating thread %d \n",  i);
     }

     
     void *rslt;
     for(int i = 0; i < _numThreads; i++) {
       stat = pthread_join(_thread[i], &rslt);
       if(stat != 0) printf("  err joining thread %d \n",  i);
     }
     
  }

  template <class VarTYPE>
  void
  MultiThreadedTopK<VarTYPE>::uncompress(               
     int shrinkedSize // size of the compressed ("source") vector
   , std::vector< PairIndexValue <VarTYPE> > & glbShrinkedVect  // source vect
   , std::unique_ptr<VarTYPE []>  & vector  )  // destination vector
  {
     _shrinkedSize = shrinkedSize;
     _pPairsVect = glbShrinkedVect.data();  
     _destVector = vector.get(); 

     //...

      for(int i=0;  i < _numThreads; i++) setThreadParameters_uncompress(i);

      int stat;
      for(int i = 0; i < _numThreads; i++) {
        stat = pthread_create( &(_thread[i]), NULL, 
	          &(thread_func_compressTopK<VarTYPE>), 
		  (void *) &(_pThreadParams_uncompress[i]));

          if(stat != 0) printf(" err creating thread %d \n",  i);
      }

       void *rslt;
       for(int i = 0; i < _numThreads; i++) {
         stat = pthread_join(_thread[i], &rslt);
         if(stat != 0) printf("  err joining thread %d \n",  i);
       }
  } // uncompress


} // namespace compressed_exchange
