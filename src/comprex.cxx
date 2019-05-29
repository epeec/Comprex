/*
 *  comprex.cxx
 *
 *
 */

#include <comprex.hxx>

#ifndef COMPREX_CXX
#define COMPREX_CXX

namespace compressed_exchange {

  template <class VarTYPE>
  ComprEx<VarTYPE>::ComprEx( 
                gaspi::Runtime & runTime
              , gaspi::Context & context
	      , gaspi::segment::Segment & segment
	      , int origSize   // size of the vectors to work with
	      ) 
        :
       _gpiCxx_runtime(runTime)
     , _gpiCxx_context(context)
     , _gpiCxx_segment(segment)
     , _origSize(origSize)
     , _origVector() 
     , _restsVector()
     , _treshold(0)
     , _shrinkedSize(0)
     , _auxInfoVectSize(0)
     , _shrinkedVect()
     , _auxInfoVectr()
     , _buffSizeBytes(0)
   {
      _restsVector =  std::unique_ptr< VarTYPE[] > (new VarTYPE [_origSize]);
      flushTheRestsVector();
   }
   
   template <class VarTYPE>
   ComprEx<VarTYPE>::~ComprEx()
   {

   }

   template <class VarTYPE>
   void ComprEx<VarTYPE>::flushTheRestsVector()
   {
      std::memset(static_cast<void *> (_restsVector.get()), 0, 
                                                   _origSize*sizeof(VarTYPE));
   }
   
   template <class VarTYPE>
   const VarTYPE * ComprEx<VarTYPE>::entryPointerRestsVector() const
   {
     return _restsVector.get();
   }

   template <class VarTYPE>
   bool ComprEx<VarTYPE>::checkFulfilled_forItem(int i)
   {
     _restsVector[i] += _origVector[i];
     if(std::abs(_restsVector[i]) >= _treshold) return true;
     return false;
   }

   template <class VarTYPE>
   void ComprEx<VarTYPE>::printCompressedVector(const char *fullPath) const
   {

     const int rank = static_cast<int> ( _gpiCxx_context.rank().get() ); 

     char s_fName[160];
     sprintf(s_fName, "%s/compressedVector.%d", fullPath, rank);
     std::ofstream file;
     file.open(s_fName);

     // typedef typename std::vector<VarTYPE>::iterator MyIterType;
     // NB!!  typedef can not be a template !! Therefore:
     //template <class VarTYPE>
     //using MyIterType = typename std::vector<VarTYPE>::const_iterator;
     // but should be defined outside of the function.

     // size of the run-lengths sequence 
     file << "compressed-vector size: " << _shrinkedSize << std::endl;
     int cntr = 0;
     for ( typename std::vector<VarTYPE>::const_iterator it = _shrinkedVect.begin(); 
	  it != _shrinkedVect.end(); ++it) {
       file << " compressedVector[ " << cntr << " ] = " 
	    << *it 
            << std::endl;
       cntr++;  
     }//  
     // if(cntr != _shrinkedSize) throw()


     file << " == Once again, using std::vector.data()-ptr  == " << std::endl;
     const VarTYPE *myPtr = _shrinkedVect.data();
     for(int i = 0; i < _shrinkedSize; i++) {
       file << " again compressedVector[ " << i << " ] = " 
	    << *(myPtr+i)
            << std::endl;       
     }
     
     file.close();
     
   } // printCompressedVector



   template <class VarTYPE>
   void ComprEx<VarTYPE>::printOriginalVector(const char *fullPath)  const
   {

   }
 
   template <class VarTYPE>
   const VarTYPE* 
   ComprEx<VarTYPE>::getPtrShrinkedVector() const
   {
   
      return  _shrinkedVect.data();
   }

   template <class VarTYPE>
   const int* 
   ComprEx<VarTYPE>::getPtrAuxiliaryInfoVector() const
   {

      return _auxInfoVectr.data();
   }

   template <class VarTYPE>
   const int
   ComprEx<VarTYPE>:: getSizeCompressedVector() const
   {
      return _shrinkedSize;
   }

   template <class VarTYPE>
   const int 
   ComprEx<VarTYPE>::getSizeAuxiliaryInfoVector() const
   {
      return _auxInfoVectSize;
   }

   template <class VarTYPE>
   const VarTYPE 
   ComprEx<VarTYPE>::getTreshold() const
   {
      return _treshold;
   }

   template <class VarTYPE>
   void 
   ComprEx<VarTYPE>::erazeTheLocalStructures()
   {
     _shrinkedVect.clear();
     _shrinkedSize = 0;

     _auxInfoVectr.clear(); 
     _auxInfoVectSize = 0;
   }

   template <class VarTYPE>
   void 
   ComprEx<VarTYPE>::communicateDataBufferSize_senderSide(
                                            gaspi::group::Rank  destRank
					    , int tag  )
   {
       gaspi::singlesided::write::SourceBuffer
                    srcBuff( _gpiCxx_segment, sizeof(int)) ;

       srcBuff.connectToRemoteTarget(_gpiCxx_context,
                                     destRank,
                                     tag).waitForCompletion();

       int* buffEntry = (reinterpret_cast<int *>(srcBuff.address()));
       *buffEntry =  _buffSizeBytes;
          
       srcBuff.initTransfer(_gpiCxx_context);
   }
   
   template <class VarTYPE>
   void 
   ComprEx<VarTYPE>::communicateDataBufferSize_recverSide(
                                              gaspi::group::Rank  srcRank
					      , int tag )
   {
       gaspi::singlesided::write::TargetBuffer
                    targBuff( _gpiCxx_segment, sizeof(int)) ;

       targBuff.connectToRemoteSource(_gpiCxx_context,
                                        srcRank,
                                        tag).waitForCompletion();

       targBuff.waitForCompletion(); 

       int* buffEntry = (reinterpret_cast<int *>(targBuff.address()));
       _buffSizeBytes = *buffEntry;

  }

   template <class VarTYPE>
   void 
   ComprEx<VarTYPE>::sendCompressedVectorToDestRank(
	            gaspi::group::Rank  destRank // destination rank 
		  , int tag)
   {

     calculateSizeSourceBuffer();
     
     // first communicate (send to dest rank) the buffer size
     communicateDataBufferSize_senderSide(destRank, tag);

     // then communicate the data itself 
     // alloc GaspiCxx source  buffer
     gaspi::singlesided::write::SourceBuffer
           srcBuff(_gpiCxx_segment, _buffSizeBytes) ;

     srcBuff.connectToRemoteTarget(_gpiCxx_context,
                                     destRank,
                                     tag).waitForCompletion();
     fillInSourceBuffer(srcBuff);

     srcBuff.initTransfer(_gpiCxx_context);
    
   } //sendCompressedVectorToDestRank


   template <class VarTYPE>
   void 
   ComprEx<VarTYPE>::getCompressedVectorFromSrcRank(
                    std::unique_ptr<VarTYPE []>  & vector 
	          , gaspi::group::Rank  srcRank // source rank              
		  , int tag)
   {

     // first communicate (get from src rank) the buffer size
     communicateDataBufferSize_recverSide(srcRank, tag);

     
     // then get  the data itself 
     // alloc GaspiCxx  target- buffer
     gaspi::singlesided::write::TargetBuffer 
           targetBuff(_gpiCxx_segment, _buffSizeBytes);

      targetBuff.connectToRemoteSource(_gpiCxx_context,
                                        srcRank,
                                        tag).waitForCompletion();
      targetBuff.waitForCompletion();

      //deCompressVector(targetBuff, vector);
      deCompressVector_inLocalStructs(targetBuff);
      fillInVectorFromLocalStructs(vector);

   } // getCompressedVectorFromSrcRank

   template <class VarTYPE>
   void 
   ComprEx<VarTYPE>::compress_and_p2pVectorWriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
	       , VarTYPE treshold            // treshold
	       , gaspi::group::Rank  destRank// destination rank
               , int tag                     // message tag
               , int nThreads)   // number of threads used in compression
   {

     if(nThreads == 1) {
          compressVectorSingleThreaded(vector, treshold);
     }
     else if(nThreads > 1 ) {
          

     }
     else {
       throw std::runtime_error (
            "For this number of threads RLE compression not implemented ..");
     }
     sendCompressedVectorToDestRank(destRank, tag);

   } // compress_and_writeRemote

    
   template <class VarTYPE>
   void 
   ComprEx<VarTYPE>::p2pVectorGetRemote(
	       std::unique_ptr<VarTYPE []>  & vector // pointer to dest vector
	       , gaspi::group::Rank  srcRank// source rank
               , int tag                    // message tag
               , int nThreads)
   {

     getCompressedVectorFromSrcRank(vector, srcRank, tag);
     erazeTheLocalStructures();
   } //  p2pGetRemoteVector

} // end namespace compressed_exchange

#endif //#define COMPREX_CXX
