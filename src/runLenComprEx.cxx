// runLengthComprEx.cxx -> run-length-encoding compression and exchange, impl

//#include <runLenCompEx.hxx>
#include <comprex.hxx>

#ifndef RUN_LEN_COMPR_EX_CXX
#define RUN_LEN_COMPR_EX_CXX

namespace compressed_exchange {
   
  template <class VarTYPE>
  ComprExRunLengths<VarTYPE>::ComprExRunLengths(
                                    gaspi::Runtime & runTime
                                  , gaspi::Context & context
	                          , gaspi::segment::Segment & segment                                             )
      :
      _signumFlag(0)
    , _mThrLRE()   
    , ComprEx<VarTYPE>(runTime, context, segment)
  {

  } 

  template <class VarTYPE>
  ComprExRunLengths<VarTYPE>::~ComprExRunLengths()
  {

  } 

   template <class VarTYPE>
   void 
   ComprExRunLengths<VarTYPE>::printCompressedVector_inOriginalSize(
                                                 const char *fullPath) const
   {
     const int rank = static_cast<int> ( 
                      ComprEx<VarTYPE>::_gpiCxx_context.rank().get() ); 

     char s_fName[160];
     sprintf(s_fName, "%s/compressedVector_origSz.%d", fullPath, rank);
     std::ofstream file;
     file.open(s_fName);

     int cntr_orig = 0;
     int cntr_comprs = 0;
     int cntr_runLengths = 0;

     if(_signumFlag == 0) {  // strting with "no"-sequence
       cntr_orig += ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths];
       for(int i = 0; i < cntr_orig; i++) {
         file << " compressdVect_origSz[ " << i << " ] = "
              << 0 << " (empty item)" << std::endl;
       }
       cntr_runLengths++;
     } // if(_signumFlag == 0)

     // now alternate (i) "yes"-items (ii) "no" items
     while (cntr_orig <  ComprEx<VarTYPE>::_origSize) {

       // "yes"-items
       for(int i = cntr_orig;
	   i < cntr_orig + ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths]; 
           i++) {

         //if(cntr_comprs == _shrinkedSize) throw;
         file << " compressdVect_origSz[ " << i << " ] = "
              << ComprEx<VarTYPE>::_shrinkedVect[cntr_comprs] <<  std::endl;
         cntr_comprs++;
       }
       cntr_orig += ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths];
       cntr_runLengths++;

       if(cntr_orig == ComprEx<VarTYPE>::_origSize) break;
          
        // "no"-items
       for(int i = cntr_orig;
	   i < cntr_orig + ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths]; 
           i++) {

         file << " compressdVect_origSz[ " << i << " ] = "
              << 0 << " (empty item)" << std::endl;
       }
      cntr_orig += ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths];
      cntr_runLengths++;

     } // while 

     file.close();
   }

   template <class VarTYPE>
   void 
   ComprExRunLengths<VarTYPE>::fillInVectorFromLocalStructs(
                                      std::unique_ptr<VarTYPE []> & vector)
   {

     int cntr_orig = 0;
     int cntr_comprs = 0;
     int cntr_runLengths = 0;

     if(_signumFlag == 0) {  // strting with "no"-sequence
       cntr_orig += ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths];
       for(int i = 0; i < cntr_orig; i++) {
         vector[i] = 0;
       }
       cntr_runLengths++;
     } // if(_signumFlag == 0)

     // now alternate (i) "yes"-items (ii) "no" items
     while (cntr_orig <  ComprEx<VarTYPE>::_origSize) {

       // "yes"-items
       for(int i = cntr_orig;
	   i < cntr_orig + ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths]; 
           i++) {

         //if(cntr_comprs == _shrinkedSize) throw;
         vector[i] = ComprEx<VarTYPE>::_shrinkedVect[cntr_comprs];
         cntr_comprs++;
       }
       cntr_orig += ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths];
       cntr_runLengths++;

       if(cntr_orig == ComprEx<VarTYPE>::_origSize) break;
          
        // "no"-items
       for(int i = cntr_orig;
	   i < cntr_orig + ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths]; 
           i++) {
         vector[i] = 0;       
       }
      cntr_orig += ComprEx<VarTYPE>::_auxInfoVectr[cntr_runLengths];
      cntr_runLengths++;

     } // while 

   }//fillInVectorFromLocalStructs


   template <class VarTYPE>
   void 
   ComprExRunLengths<VarTYPE>::printAuxiliaryInfoVector(
                                           const char *fullPath) const
   {
     const int rank = 
      static_cast<int> ( ComprEx<VarTYPE>::_gpiCxx_context.rank().get() ); 

     char s_fName[160];
     sprintf(s_fName, "%s/runLengths.%d", fullPath, rank);
     std::ofstream file;
     file.open(s_fName);

     typedef typename std::vector<int>::const_iterator MyIterType;
     

     // start with the signum flag
     file << "signumFlag: " << _signumFlag << std::endl;
     // size of the run-lengths sequence 
     file << "run-lengths-sequence ttl size: " 
           << ComprEx<VarTYPE>::_auxInfoVectSize << std::endl;

     // now the run-length sequence
     int cntr = 0;    
     for ( MyIterType it = ComprEx<VarTYPE>::_auxInfoVectr.begin(); 
	  it != ComprEx<VarTYPE>::_auxInfoVectr.end(); ++it) {
       file << " runLength[ " << cntr << " ] = " 
            <<  ComprEx<VarTYPE>::_auxInfoVectr[cntr]  //*it 
            << std::endl;
       cntr++;  
     }//  
     
     file.close();
   }


   template <class VarTYPE>
   const int 
   ComprExRunLengths<VarTYPE>::getSignumFlag() const
   {
      return _signumFlag;
   }
  
   template <class VarTYPE>
   void 
   ComprExRunLengths<VarTYPE>::compressVectorSingleThreaded(
           std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
         , int size                    // vector´s (original) size
	 , VarTYPE treshold)      // treshold
   {
     ComprEx<VarTYPE>::_origVector = vector.get();
     ComprEx<VarTYPE>::_origSize = size;
     ComprEx<VarTYPE>::_treshold = treshold;

     ComprEx<VarTYPE>::_shrinkedSize = 0;
     ComprEx<VarTYPE>::_auxInfoVectSize = 0;

     // a check-counter to control the vector length
     // increase it each time _auxInfoVectr.push_back(..) is executed
     int check_cntr = 0;

     // current run-length countrers for "yes" and "no" items    
     int crrRunLength_yes = 0;
     int crrRunLength_no = 0;
     // check the first vector item, set the signum-flag
     if(std::abs(ComprEx<VarTYPE>::_origVector[0]) >= treshold) {
       //_shrinkedVect[_shrinkedSize] = _origVector[0];  // store the 0-th item
       ComprEx<VarTYPE>::_shrinkedVect.push_back(
                              ComprEx<VarTYPE>::_origVector[0]);
       ComprEx<VarTYPE>::_shrinkedSize++; // increase the shrinkedSize-counter
       crrRunLength_yes++;    // increase the "yes"-counter   
       _signumFlag = 1;
     }
     else {  //if(_origVect[0] < treshold)
       // The convention: the vector _auxInfoVectr[] always starts
       // with the (number of) runs of meaningful values 
       // (i.e. the number of "yes"-runs) even if it is zero.
       // The latter is exactly the case here, thus 
       // increase the  crrRunLength_no-counter 
       crrRunLength_no++;
       _signumFlag = 0;
     }

     for(int i = 1; i < ComprEx<VarTYPE>::_origSize; i++) {
       if(std::abs(ComprEx<VarTYPE>::_origVector[i]) >= treshold) {
         //_shrinkedVect[_shrinkedSize] = _origVector[i];
	 ComprEx<VarTYPE>::_shrinkedVect.push_back(
                                 ComprEx<VarTYPE>::_origVector[i]); 
         ComprEx<VarTYPE>::_shrinkedSize++;
       
         if(crrRunLength_yes >= 0) { // the previous number was an "yes"-number
	   crrRunLength_yes++;         // Then just increase the counter
         }
         if(crrRunLength_no > 0) {  //the previous number was a "no"-number 
           // write the current "no"-length in _auxInfoVectr[] 
           // and increase the counter 
	   //_auxInfoVectr[_auxInfoVectSize] =  crrRunLength_no; 
           ComprEx<VarTYPE>::_auxInfoVectr.push_back(crrRunLength_no);
           ComprEx<VarTYPE>::_auxInfoVectSize++;
           //increase the check-counter (evntl. optional)
           check_cntr += crrRunLength_no;
           // set crrRunLength_no to ZERO
           crrRunLength_no = 0;
         } // if(crrRunLength_no > 0)
         
       } //if(_origVector[i] >= treshold)
       else {  // i.e. _origVector[i] < treshold -> write nothing in _shrinkedVect[]
         if(crrRunLength_no >= 0) {    // the previous number was a "no"-number
	   crrRunLength_no++;         // Then just increase the counter
         }
         if(crrRunLength_yes > 0) {//the previous number was a "yes"-number 
           // write the current "yes"-length in _auxInfoVectr[] 
           // (it should be != 0) and increase the counter 
	   //_auxInfoVectr[_auxInfoVectSize] =  crrRunLength_yes; 
           ComprEx<VarTYPE>::_auxInfoVectr.push_back(crrRunLength_yes); 
           ComprEx<VarTYPE>::_auxInfoVectSize++;
           //increase the check-counter (evntl. optional)
           check_cntr += crrRunLength_yes;
           // set crrRunLength_yes to ZERO
           crrRunLength_yes=0;
         } //if(crrRunLength_yes > 0)

       } // else .. if(_origVector[i] >= treshold)

     } //for(int i  

     // push_back the last (still not stored) sequence
     if( (crrRunLength_yes > 0) && (crrRunLength_no == 0)) {
           ComprEx<VarTYPE>::_auxInfoVectr.push_back(crrRunLength_yes); 
           ComprEx<VarTYPE>::_auxInfoVectSize++;
           //increase the check-counter (evntl. optional)
           check_cntr += crrRunLength_yes;
     }
     if( (crrRunLength_no > 0) && (crrRunLength_yes == 0)) {
           ComprEx<VarTYPE>::_auxInfoVectr.push_back(crrRunLength_no); 
           ComprEx<VarTYPE>::_auxInfoVectSize++;
           //increase the check-counter (evntl. optional)
           check_cntr += crrRunLength_no;
     }
     
     // check here if the sum of _auxInfoVectr[]-items 
     // equals the original size, if not -> throw
     // May leave this check, it is not "expensive"
     if(check_cntr != ComprEx<VarTYPE>::_origSize) {
       throw std::runtime_error (
            "The sum of run-length-vector entries NOT equal orig. vect. size");
     }

   } // compressVectorSingleThreaded
  


   template <class VarTYPE>
   void 
   ComprExRunLengths<VarTYPE>::calculateSizeSourceBuffer()
   {
     ComprEx<VarTYPE>::_buffSizeBytes = 0;

     // start with 4 integers:
     //    int original Vector Size
     //    int shrinkedVectorSize
     //    int runLengthSize
     //    int signumFlag
     ComprEx<VarTYPE>::_buffSizeBytes += 4*sizeof(int);

     // then the runLengths[] array, i.e. #_auxInfoVectSize integers
    ComprEx<VarTYPE>::_buffSizeBytes += 
               ComprEx<VarTYPE>::_auxInfoVectSize * sizeof(int);

     // then the compressed vector values, i.e. #_shrinkedSize VarTYPE-s
     ComprEx<VarTYPE>::_buffSizeBytes += 
                   ComprEx<VarTYPE>::_shrinkedSize * sizeof(VarTYPE);

   }

   template <class VarTYPE>
   void 
   ComprExRunLengths<VarTYPE>::fillInSourceBuffer(
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

       //  run-Length-vector Size, int
       *(reinterpret_cast<int *>(pCrr)) = ComprEx<VarTYPE>::_auxInfoVectSize;
       pCrr += sizeof(int);
 
       // signumFlag, int
       *(reinterpret_cast<int *>(pCrr)) = _signumFlag;
       pCrr += sizeof(int);

       // _runLengthsSize integers, the contents of _auxInfoVectr[]-vector
       std::memcpy(pCrr, ComprEx<VarTYPE>::_auxInfoVectr.data(), 
                     ComprEx<VarTYPE>::_auxInfoVectSize*sizeof(int) ); 
       pCrr +=  ComprEx<VarTYPE>::_auxInfoVectSize*sizeof(int);

       // _shrinkedSize VarTYPE, the contents of _shrinkedVect[]-vector
       std::memcpy(pCrr, ComprEx<VarTYPE>::_shrinkedVect.data(), 
                       ComprEx<VarTYPE>::_shrinkedSize*sizeof(VarTYPE) ); 
       pCrr +=  ComprEx<VarTYPE>::_shrinkedSize*sizeof(VarTYPE);

   }// fillInSourceBuffer

   template <class VarTYPE>
   void 
   ComprExRunLengths<VarTYPE>::deCompressVector_inLocalStructs(
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

       //  run-Length-vector Size, int
       ComprEx<VarTYPE>::_auxInfoVectSize = *(reinterpret_cast<int *>(pCrr));
       pCrr += sizeof(int);
 
       // signumFlag, int
       _signumFlag = *(reinterpret_cast<int *>(pCrr));
       pCrr += sizeof(int);

       
       // _runLengthsSize integers, the contents of _auxInfoVectr[]-vector
       ComprEx<VarTYPE>::_auxInfoVectr.resize(ComprEx<VarTYPE>::_auxInfoVectSize);
       std::memcpy(ComprEx<VarTYPE>::_auxInfoVectr.data(), pCrr, 
                       ComprEx<VarTYPE>::_auxInfoVectSize*sizeof(int) ); 
       pCrr +=  ComprEx<VarTYPE>::_auxInfoVectSize*sizeof(int);

       // _shrinkedSize VarTYPE, the contents of _shrinkedVect[]-vector
       ComprEx<VarTYPE>::_shrinkedVect.resize(ComprEx<VarTYPE>::_shrinkedSize); 
       std::memcpy(ComprEx<VarTYPE>::_shrinkedVect.data(), pCrr, 
                  ComprEx<VarTYPE>::_shrinkedSize*sizeof(VarTYPE) ); 
       pCrr +=  ComprEx<VarTYPE>::_shrinkedSize*sizeof(VarTYPE);
       

   } // deCompressVector_inLocalStructs
   

   template <class VarTYPE>
   void 
   ComprExRunLengths<VarTYPE>::compress_and_p2pVectorWriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
               , int size                    // vector´s (original) size
	       , VarTYPE treshold            // treshold
	       , gaspi::group::Rank  destRank// destination rank
               , int tag                     // message tag
               , int nThreads)   // number of threads used in compression
   {

     if(nThreads == 1) {
          compressVectorSingleThreaded(vector, size, treshold);
     }
     else if(nThreads > 1 ) {
          _mThrLRE = std::unique_ptr<MultiThreadedRLE<VarTYPE> > 
	    (new MultiThreadedRLE<VarTYPE> (
		     this->_origSize
		   , ComprEx<VarTYPE>::_origVector
		   , ComprEx<VarTYPE>::_treshold
		   , _signumFlag
	           , ComprEx<VarTYPE>::_shrinkedVect
	      	   , ComprEx<VarTYPE>::_auxInfoVectr
	     	   , nThreads)
            );  
	  _mThrLRE->compress(ComprEx<VarTYPE>::_shrinkedSize, 
                             ComprEx<VarTYPE>::_auxInfoVectSize);
     }
     else {
       throw std::runtime_error (
            "For this number of threads RLE compression not implemented ..");
     }
     ComprEx<VarTYPE>::sendCompressedVectorToDestRank(destRank, tag);
     ComprEx<VarTYPE>::erazeTheLocalStructures();

   } // compress_and_writeRemote

} // namespace compressed_exchange

#endif
