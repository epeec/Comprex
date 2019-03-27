/*
 *  comprex.cxx
 *
 *
 */

#include <comprex.hxx>

#ifndef COMPREX_CXX
#define COMPREX_CXX

namespace compressed_exchange {

  MyClass::MyClass(std::string myString)
    :
    _myMsg(myString)
  {

  }

  MyClass::~MyClass() 
  {

  }

  void
  MyClass::printMessage() const
  {
    // a test comment added, to check how .gitignore works
    std::cout << "The message is: " << _myMsg << std::endl; 
  }



  template <class VarTYPE>
  ComprEx<VarTYPE>::ComprEx( 
	        CompressionType type
	      , gaspi::Runtime & runTime
              , gaspi::Context & context
	      , gaspi::segment::Segment & segment
	      ) 
    :
     _type(type)
     , _gpiCxx_runtime(runTime)
     , _gpiCxx_context(context)
     , _gpiCxx_segment(segment)
     , _origSize(0)
     , _origVector() 
     , _treshold(0)
     , _compresrRL()
   {

     if(_type == CompressionType::runLengthEncoding) {
         _compresrRL = std::unique_ptr<impl::CompressorRunLengths<VarTYPE> > (
		     new impl::CompressorRunLengths<VarTYPE>(
                                 _gpiCxx_runtime
			       , _gpiCxx_context
                               , _gpiCxx_segment
                                                            ) 
                                                                          );
     }
     if(_type == CompressionType::sparseIndexing) {
 
     }

   }

   
   template <class VarTYPE>
   ComprEx<VarTYPE>::~ComprEx()
   {

   }

   template <class VarTYPE>
   void ComprEx<VarTYPE>::compress_and_p2pWriteRemote(
 	       std::unique_ptr<VarTYPE []> const & vector//pointer to the vector
	       //const VarTYPE * pVector     
               , int size                         // vector´s (original) size
	       , VarTYPE treshold                 // treshold
	       , gaspi::group::Rank  srcRank      // source rank
	       , gaspi::group::Rank  destRank     // destination rank
               , int nThreads) // number of threads used in compression
   {
 
     _origVector = vector.get();
     //_origVector = pVector;
     _origSize = size;
     _treshold = treshold;

     if(_type == CompressionType::runLengthEncoding) {
       //    _compresrRL->compress_and_p2pWriteRemote(
       //		 vector, size, treshold, srcRank, destRank, nThreads);
     }
     if(_type == CompressionType::sparseIndexing) {
       //
     }

     //int myRnk = static_cast<int> ( _gpiCxx_context.rank().get() ); 
     //for(int i = 0; i < size; i++) {
     //  printf("\n [%d] ref to myVect[%d]:%e ", 
     //           myRnk, i,_origVector[i]);// vector[i]);
     //}


   } //compress_and_WriteRemote 

   template <class VarTYPE>
   void ComprEx<VarTYPE>::printCompressedVector(const char *fullPath) const
   {
     if(_type == CompressionType::runLengthEncoding) {
       _compresrRL->printCompressedVector(fullPath);                                                                           
     }
     if(_type == CompressionType::sparseIndexing) {
 
     }
     
   }

   template <class VarTYPE>
   void ComprEx<VarTYPE>::printCompressedVector_inOriginalSize(
                                            const char *fullPath) const
   {

     if(_type == CompressionType::runLengthEncoding) {
       _compresrRL->printCompressedVector_inOriginalSize(fullPath);                                                                           
     }
     if(_type == CompressionType::sparseIndexing) {
 
     }

   }

   template <class VarTYPE>
   void ComprEx<VarTYPE>::printRunLengthsVector(const char *fullPath) const
   {

     if(_type == CompressionType::runLengthEncoding) {
       _compresrRL->printRunLengthsVector(fullPath);                                                                           
     }
     if(_type == CompressionType::sparseIndexing) {
 
     }

    }

   template <class VarTYPE>
   void ComprEx<VarTYPE>::printOriginalVector(const char *fullPath)  const
   {

   }
 

} // end namespace compressed_exchange

#endif //#define COMPREX_CXX
