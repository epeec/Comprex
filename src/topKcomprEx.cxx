// topKcomprEx.cxx -> communicate the first top-K-percents of the items having 
// highest absolute values

#include <comprex.hxx>

#ifndef TOP_K_COMPR_EX_CXX
#define TOP_K_COMPR_EX_CXX

namespace compressed_exchange {

  template <class VarTYPE>
  ComprExTopK<VarTYPE>::ComprExTopK(
                                    gaspi::Runtime & runTime
                                  , gaspi::Context & context
	                          , gaspi::segment::Segment & segment
	                          , int origSize   // size of the vectors to work with
                                               )
      :
    //_rawVectPairs()
    _crrWriteCall(0)
    , _vectPairs()   
    , ComprEx<VarTYPE>(runTime, context, segment, origSize)
  {

  } 

  template <class VarTYPE>
  ComprExTopK<VarTYPE>::~ComprExTopK()
  {

  } 

  
  template <class VarTYPE>
  void ComprExTopK<VarTYPE>::printAuxiliaryInfoVector( 
                                           const char *fullPath) const
  {
     const int rank = 
      static_cast<int> ( ComprEx<VarTYPE>::_gpiCxx_context.rank().get() ); 

     int itr = _crrWriteCall;

     char s_fName[160];
     sprintf(s_fName, "%s/topK_vect_sorted_itr%d.%d", fullPath, itr, rank);
     std::ofstream file;
     file.open(s_fName);

     typedef typename std::vector<PairIndexValue<VarTYPE> >::const_iterator 
                                                                   MyIterType;     

     file << "original vector size: " << ComprEx<VarTYPE>::_origSize 
	  << " pairs-vector size: "<< _vectPairs.size()   << std::endl;
     file << " top-K-percents of items to send:" << ComprEx<VarTYPE>::_treshold
          << " last-index-to-be-sent: " << ComprEx<VarTYPE>::_shrinkedSize-1 << std::endl;
     // now the run-length sequence
     int cntr = 0;    
     for ( MyIterType it = _vectPairs.begin(); 
	  it != _vectPairs.end(); ++it) {
       file << " sortedVect[" << cntr << "], ->origIdx: " 
            << it->idx << " , ->orig_val = " 
            << it->val  //*it 
            << std::endl;
       cntr++;  
     }//  
     
     file.close();
  } //printAuxiliaryInfoVector  

  template <class VarTYPE>
  const struct PairIndexValue<VarTYPE> *
  ComprExTopK<VarTYPE>::entryPointerVectorPairs() const
  {
    return _vectPairs.data();
  }

  template <class VarTYPE>
  void ComprExTopK<VarTYPE>::calculateNumberOfItemsToSend(VarTYPE topK_percents)
  {

     // store the value of K inComprEx<VarTYPE>::_threshold
     ComprEx<VarTYPE>::_treshold = topK_percents; 

     // store the number of itzems to be send (from the sorted vector) in _shrinkedSize
     ComprEx<VarTYPE>::_shrinkedSize = static_cast<int> (
	(static_cast<double> (topK_percents))/100.0  * 
        static_cast<double> (ComprEx<VarTYPE>::_origSize)
           );
     // the last item to send should have an index   #(_shrinkedSize-1) 
     
  }


  template <class VarTYPE>
  void ComprExTopK<VarTYPE>::fillIn_absValuesVector_and_sort()
  {

       _vectPairs.resize(ComprEx<VarTYPE>::_origSize);

       for(int i = 0; i < ComprEx<VarTYPE>::_origSize; i++) {
	 ComprEx<VarTYPE>::_restsVector[i] += ComprEx<VarTYPE>::_origVector[i];
         _vectPairs[i].idx = i;
         _vectPairs[i].val = ComprEx<VarTYPE>::_restsVector[i];
       }
       
       std::sort(_vectPairs.begin(), _vectPairs.end(), 
                                                    sortPredicate_absValue);

  }  

  
  template <class VarTYPE>
  void ComprExTopK<VarTYPE>::calculateSizeSourceBuffer()
  {
    ComprEx<VarTYPE>::_buffSizeBytes = 0;

    // start with 2 integers:
    //    int original vector size
    //    int shrinked vector size
    ComprEx<VarTYPE>::_buffSizeBytes += 2*sizeof(int);

    // ComprEx<VarTYPE>::_shrinkedSize-items of PairIndexValue-type 
    ComprEx<VarTYPE>::_buffSizeBytes += 
      ComprEx<VarTYPE>::_shrinkedSize * sizeof(struct PairIndexValue<VarTYPE>);

  }  

   template <class VarTYPE>
   void 
   ComprExTopK<VarTYPE>::fillInSourceBuffer(
           gaspi::singlesided::write::SourceBuffer &srcBuff)
   {
       void* pBuffBegin = srcBuff.address();
       void* pCrr =   pBuffBegin;
       
       // originalVectorSize , int
       *(reinterpret_cast<int *>(pCrr)) = ComprEx<VarTYPE>::_origSize;
       pCrr += sizeof(int);

       // shrinkedVectorSize , int
       *(reinterpret_cast<int *>(pCrr)) = ComprEx<VarTYPE>::_shrinkedSize;
       pCrr += sizeof(int);

       // ComprEx<VarTYPE>::_shrinkedSize-items of PairIndexValue-type 
       std::memcpy(pCrr, _vectPairs.data(), 
         ComprEx<VarTYPE>::_shrinkedSize*sizeof(struct PairIndexValue<VarTYPE>) ); 
       pCrr += 
         ComprEx<VarTYPE>::_shrinkedSize*sizeof(struct PairIndexValue<VarTYPE>); 

   }// fillInSourceBuffer

  template <class VarTYPE>
  void ComprExTopK<VarTYPE>::compressVectorSingleThreaded(
           std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
	 , VarTYPE topK_percents)      // the value of K
   {
     ComprEx<VarTYPE>::_origVector = vector.get();

     // calculate the number of items to send in the sorted vector
     calculateNumberOfItemsToSend(topK_percents);

     fillIn_absValuesVector_and_sort();
     //calculateSizeSourceBuffer();// call it in sendCompressedVectorToDestRank(..)

   } // compressVectorSingleThreaded


   template <class VarTYPE>
   void  ComprExTopK<VarTYPE>::inRestsVectorSetToZeroTheItemsJustSent()
   {

     int idx; 
     for(int i = 0; i < ComprEx<VarTYPE>::_shrinkedSize; i++) {
       ComprEx<VarTYPE>::_restsVector[_vectPairs[i].idx] = 0;
     }

   }    

   template <class VarTYPE>
   void  ComprExTopK<VarTYPE>::deCompressVector_inLocalStructs(
		       gaspi::singlesided::write::TargetBuffer &targBuff)
   {
       void* pBuffBegin = targBuff.address();
       void* pCrr =   pBuffBegin;

       // originalVectorSize , int
       if(*(reinterpret_cast<int *>(pCrr)) != ComprEx<VarTYPE>::_origSize) {
	 printf("\n [%d] sendr-rank-orig-size:%d my-orig_size:%d \n",
		ComprEx<VarTYPE>::_gpiCxx_context.rank().get(), 
                *(reinterpret_cast<int *>(pCrr)), ComprEx<VarTYPE>::_origSize);
           throw std::runtime_error (
            "sender-rank original-vector-size differs than the receivers one.");
       }
       //_origSize = *(reinterpret_cast<int *>(pCrr));
       pCrr += sizeof(int);
       
       // shrinkedVectorSize , int
       ComprEx<VarTYPE>::_shrinkedSize = *(reinterpret_cast<int *>(pCrr));
       pCrr += sizeof(int);

       // ComprEx<VarTYPE>::_shrinkedSize-items of PairIndexValue-type 
       _vectPairs.resize(ComprEx<VarTYPE>::_shrinkedSize);

       std::memcpy(_vectPairs.data(), pCrr, 
         ComprEx<VarTYPE>::_shrinkedSize*sizeof(struct PairIndexValue<VarTYPE>) ); 
       pCrr += 
         ComprEx<VarTYPE>::_shrinkedSize*sizeof(struct PairIndexValue<VarTYPE>); 

   } // deCompressVector_inLocalStructs


   template <class VarTYPE>
   void ComprExTopK<VarTYPE>::fillInVectorFromLocalStructs(
                                      std::unique_ptr<VarTYPE []> & vector)
   {
     int idx; 
     for(int i = 0; i < ComprEx<VarTYPE>::_shrinkedSize; i++) {
       vector[_vectPairs[i].idx] = _vectPairs[i].val;
     }
       
   }

  template <class VarTYPE>
  void ComprExTopK<VarTYPE>::compress_and_p2pVectorWriteRemote(
        std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
      , VarTYPE topK_percents  // top K-percents of the items to be transferred
      , gaspi::group::Rank  destRank// destination rank
      , int tag                     // message tag
      , int nThreads)
   {

     if((topK_percents < 0) || (topK_percents > 100)) {
        throw  std::runtime_error (
            " the value of topK-percents must be within the interval [0, 100]"); 
     }
     
     if(nThreads == 1) {
        compressVectorSingleThreaded(vector, topK_percents);
     }
     else if(nThreads > 1 ) {
       //...
     }
     else {
       throw std::runtime_error (
            "For this number of threads top-K-percents compression not implemented ..");
     }
     ComprEx<VarTYPE>::sendCompressedVectorToDestRank(destRank, tag);
     inRestsVectorSetToZeroTheItemsJustSent();
     //printAuxiliaryInfoVector("/scratch/stoyanov/comprEx/run");
     ComprEx<VarTYPE>::erazeTheLocalStructures();
     _crrWriteCall++;

   } // compress_and_writeRemote



}// namespace compressed_exchange

#endif
