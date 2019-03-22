/*
 *  comprex.hxx
 *
 *
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
#include <GaspiCxx/utility/ScopedAllocation.hpp>


#include <runLenCompressr.hxx>
//#include <sparseIdxCompressr.hxx>

#ifndef COMPREX_H
#define COMPREX_H

namespace compressed_exchange {

  class MyClass {
     private: 

       std::string & _myMsg;

     public:
    
      MyClass(std::string myString);
      ~MyClass();

      void printMessage() const;

  }; // myClass


  // compression type
  enum class CompressionType {
      runLengthEncoding,     // run-length-encoding (RLE) 
      sparseIndexing // sparse indexing   (SI)
  };



  template <class VarTYPE>
  class ComprEx {
       
    private:
      
       // compression type
       CompressionType  _type;

       // GaspiCxx 
       gaspi::Runtime & _gpiCxx_runtime;
       gaspi::Context & _gpiCxx_context;
       gaspi::segment::Segment & _gpiCxx_segment;
           
       // size of and a reference to the original vector
       int _origSize;
       const VarTYPE* _origVector;

       // treshold value used to compress the vector 
       VarTYPE _treshold;

       // run-lengths compress engine 
       std::unique_ptr< impl::CompressorRunLengths<VarTYPE> >  _compresrRL; 

       // sparse indexing compress engine 
       // std::unique_ptr< CompressorSparsIdx<VarTYPE> >  _compresrSI; 

    public:
       ComprEx( 
	        CompressionType type
	      , gaspi::Runtime & runTime
              , gaspi::Context & context
	      , gaspi::segment::Segment & segment
	       );

       ~ComprEx();


    // Compress the input vector of type VarTYPE and given size,
    // using the treshold provided as an argument,
    // and then GaspiCxx-send it to the destination rank destRank. 
    void compress_and_WriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
               , int size                    // vector´s (original) size
	       , VarTYPE treshold            // treshold
	       , gaspi::group::Rank  destRank// destination rank
               , int nThreads = 1);   // number of threads used in compression

    void printCompressedVector(const char *fullPath) const;
    void printCompressedVector_inOriginalSize(const char *fullPath) const;
    void printRunLengthsVector(const char *fullPath) const;
    void printOriginalVector(const char *fullPath)  const;
 
    void compressVector_RLE(
           std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
         , int size                    // vector´s (original) size
	 , VarTYPE treshold);

    void compressVector_SI(
           std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
         , int size                    // vector´s (original) size
	 , VarTYPE treshold);

    void transferCompressedVector(){};

  }; // ComprEx 
  

}  // end namespace compressed_exchange

// include the template implementation
#include <comprex.cxx>
#include <runLenCompressr.cxx>

#endif // #define COMPREX_H
