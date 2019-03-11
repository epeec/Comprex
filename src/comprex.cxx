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
  ComprEx<VarTYPE>::ComprEx( gaspi::Runtime & runTime
              , gaspi::Context & context
	      , gaspi::segment::Segment & segment
	      ) 
    :
     _gpiCxx_runtime(runTime)
     , _gpiCxx_context(context)
     ,_gpiCxx_segment(segment)
   {

   }

   
   template <class VarTYPE>
   ComprEx<VarTYPE>::~ComprEx()
   {

   }

   template <class VarTYPE>
   void ComprEx<VarTYPE>::Compress_and_WriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
               , int size                         // vector´s (original) size
	       , VarTYPE treshold                 // treshold
	       , gaspi::group::Rank  destRank)    // destination rank
   {
     int myRnk = static_cast<int> ( _gpiCxx_context.rank().get() ); 
     for(int i = 0; i < size; i++) {
       printf("\n [%d] ref to myVect[%d]:%e ", myRnk, i, vector[i]);
     }
   }

   

} // end namespace compressed_exchange

#endif //#define COMPREX_CXX
