// runLengthCompressr.cxx -> run-length-encoding compression, implementation

//#include <runLenCompressr.hxx>
#include <comprex.hxx>

#ifndef RUN_LEN_COMPRESSR_CXX
#define RUN_LEN_COMPRESSR_CXX

namespace compressed_exchange {
  namespace impl{
   
  template <class VarTYPE>
  CompressorRunLengths<VarTYPE>::CompressorRunLengths(
                                    gaspi::Runtime & runTime
                                  , gaspi::Context & context
	                          , gaspi::segment::Segment & segment                                             )
    :
     _gpiCxx_runtime(runTime)
     , _gpiCxx_context(context)
     , _gpiCxx_segment(segment)
     , _origSize(0)
     , _origVector() 
     , _treshold(0)
     , _shrinkedSize(0)
     , _shrinkedVector()
     , _runLengthSize(0)
     , _runLength()
     , _wrkVect()
     , _wrkRunLength()
  {

  } 

  template <class VarTYPE>
  CompressorRunLengths<VarTYPE>::~CompressorRunLengths()
  {

  } 


   template <class VarTYPE>
   void CompressorRunLengths<VarTYPE>::printCompressedVector(const char *fullPath) const
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
     for ( typename std::vector<VarTYPE>::const_iterator it = _wrkVect.begin(); 
	  it != _wrkVect.end(); ++it) {
       file << " compressedVector[ " << cntr << " ] = " 
	    << *it 
            << std::endl;
       cntr++;  
     }//  
     // if(cntr != _shrinkedSize) throw()


     file << " == Once again, using std::vector.data()-ptr  == " << std::endl;
     const VarTYPE *myPtr = _wrkVect.data();
     for(int i = 0; i < _shrinkedSize; i++) {
       file << " again compressedVector[ " << i << " ] = " 
	    << *(myPtr+i)
            << std::endl;       
     }
     
     file.close();
     
   }

   template <class VarTYPE>
   void CompressorRunLengths<VarTYPE>::printCompressedVector_inOriginalSize(const char *fullPath) const
   {
     const int rank = static_cast<int> ( _gpiCxx_context.rank().get() ); 

     char s_fName[160];
     sprintf(s_fName, "%s/compressedVector_origSz.%d", fullPath, rank);
     std::ofstream file;
     file.open(s_fName);

     int cntr_orig = 0;
     int cntr_comprs = 0;
     int cntr_runLengths = 0;

     if(_signumFlag == 0) {  // strting with "no"-sequence
       cntr_orig += _wrkRunLength[cntr_runLengths];
       for(int i = 0; i < cntr_orig; i++) {
         file << " compressdVect_origSz[ " << i << " ] = "
              << 0 << " (empty item)" << std::endl;
       }
       cntr_runLengths++;
     } // if(_signumFlag == 0)

     // now alternate (i) "yes"-items (ii) "no" items
     while (cntr_orig <  _origSize) {

       // "yes"-items
       for(int i = cntr_orig;
	   i < cntr_orig + _wrkRunLength[cntr_runLengths]; i++) {

         //if(cntr_comprs == _shrinkedSize) throw;
         file << " compressdVect_origSz[ " << i << " ] = "
              << _wrkVect[cntr_comprs] <<  std::endl;
         cntr_comprs++;
       }
       cntr_orig += _wrkRunLength[cntr_runLengths];
       cntr_runLengths++;

       if(cntr_orig == _origSize) break;
          
        // "no"-items
       for(int i = cntr_orig;
	   i < cntr_orig + _wrkRunLength[cntr_runLengths]; i++) {
         file << " compressdVect_origSz[ " << i << " ] = "
              << 0 << " (empty item)" << std::endl;
       }
      cntr_orig += _wrkRunLength[cntr_runLengths];
      cntr_runLengths++;

     } // while 

     file.close();
   }

   template <class VarTYPE>
   void CompressorRunLengths<VarTYPE>::printRunLengthsVector(const char *fullPath) const
   {
     const int rank = static_cast<int> ( _gpiCxx_context.rank().get() ); 

     char s_fName[160];
     sprintf(s_fName, "%s/runLengths.%d", fullPath, rank);
     std::ofstream file;
     file.open(s_fName);

     typedef typename std::vector<int>::const_iterator MyIterType;
     

     // start with the signum flag
     file << "signumFlag: " << _signumFlag << std::endl;
     // size of the run-lengths sequence 
     file << "run-lengths-sequence ttl size: " << _runLengthSize << std::endl;

     // now the run-length sequence
     int cntr = 0;    
     for ( MyIterType it = _wrkRunLength.begin(); 
	  it != _wrkRunLength.end(); ++it) {
       file << " runLength[ " << cntr << " ] = " 
            <<  _wrkRunLength[cntr]  //*it 
            << std::endl;
       cntr++;  
     }//  
     

     file.close();

   }

   template <class VarTYPE>
   void CompressorRunLengths<VarTYPE>::printOriginalVector(const char *fullPath)  const
   {

   }
  
   template <class VarTYPE>
   void CompressorRunLengths<VarTYPE>::compressVector(
           std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
         , int size                    // vector´s (original) size
	 , VarTYPE treshold)
   {
     _origVector = vector.get();
     _origSize = size;
     _treshold = treshold;

     _shrinkedSize = 0;
     _runLengthSize = 0;

     // a check-counter to control the vector length
     // increase it each time _wrkRunLength.push_back(..) is executed
     int check_cntr = 0;

     // current run-length countrers for "yes" and "no" items    
     int crrRunLength_yes = 0;
     int crrRunLength_no = 0;
     // check the first vector item, set the signum-flag
     if(_origVector[0] >= treshold) {
       //_wrkVect[_shrinkedSize] = _origVector[0];  // store the 0-th item
       _wrkVect.push_back(_origVector[0]);
       _shrinkedSize++;       // increase the shrinkedSize-counter
       crrRunLength_yes++;    // increase the "yes"-counter   
       _signumFlag = 1;
     }
     else {  //if(_origVect[0] < treshold)
       // The convention: the vector _wrkRunLength[] always starts
       // with the (number of) runs of meaningful values 
       // (i.e. the number of "yes"-runs) even if it is zero.
       // The latter is exactly the case here, thus 
       // increase the  crrRunLength_no-counter 
       crrRunLength_no++;
       _signumFlag = 0;
     }

     for(int i = 1; i < _origSize; i++) {
       if(_origVector[i] >= treshold) {
         //_wrkVect[_shrinkedSize] = _origVector[i];
	 _wrkVect.push_back(_origVector[i]); 
         _shrinkedSize++;
       
         if(crrRunLength_yes >= 0) {    // the previous number was an "yes"-number
	   crrRunLength_yes++;         // Then just increase the counter
         }
         if(crrRunLength_no > 0) {  //the previous number was a "no"-number 
           // write the current "no"-length in _wrkRunLength[] 
           // and increase the counter 
	   //_wrkRunLength[_runLengthSize] =  crrRunLength_no; 
           _wrkRunLength.push_back(crrRunLength_no);
           _runLengthSize++;
           //increase the check-counter (evntl. optional)
           check_cntr += crrRunLength_no;
           // set crrRunLength_no to ZERO
           crrRunLength_no = 0;
         } // if(crrRunLength_no > 0)
         
       } //if(_origVector[i] >= treshold)
       else {  // i.e. _origVector[i] < treshold -> write nothing in _wrkVect[]
         if(crrRunLength_no >= 0) {    // the previous number was a "no"-number
	   crrRunLength_no++;         // Then just increase the counter
         }
         if(crrRunLength_yes > 0) {//the previous number was a "yes"-number 
           // write the current "yes"-length in _wrkRunLength[] 
           // (it should be != 0) and increase the counter 
	   //_wrkRunLength[_runLengthSize] =  crrRunLength_yes; 
           _wrkRunLength.push_back(crrRunLength_yes); 
           _runLengthSize++;
           //increase the check-counter (evntl. optional)
           check_cntr += crrRunLength_yes;
           // set crrRunLength_yes to ZERO
           crrRunLength_yes=0;
         } //if(crrRunLength_yes > 0)

       } // else .. if(_origVector[i] >= treshold)

     } //for(int i  

     // push_back the last (still not stored) sequence
     if( (crrRunLength_yes > 0) && (crrRunLength_no == 0)) {
           _wrkRunLength.push_back(crrRunLength_yes); 
           _runLengthSize++;
           //increase the check-counter (evntl. optional)
           check_cntr += crrRunLength_yes;
     }
     if( (crrRunLength_no > 0) && (crrRunLength_yes == 0)) {
           _wrkRunLength.push_back(crrRunLength_no); 
           _runLengthSize++;
           //increase the check-counter (evntl. optional)
           check_cntr += crrRunLength_no;
     }
     
     // check here if the sum of _wrkRunLength[]-items 
     // equals the original size, if not -> throw
     // May leave this check, it is not "expensive"
     if(check_cntr != _origSize) {
       throw std::runtime_error (
            "The sum of run-length-vector entries NOT equal orig. vect. size");
     }


   } // compressVector_RLE

  }
} // namespace compressed_exchange

#endif
