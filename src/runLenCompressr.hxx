// runLengthCompressr.hxx -> run-length-encoding compression, definitions

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


#ifndef RUN_LENGTH_COMPRESS_H
#define RUN_LENGTH_COMPRESS_H

namespace compressed_exchange {
namespace impl {

  template <class VarTYPE>
  class CompressorRunLengths {
       
    private:

       // GaspiCxx 
       gaspi::Runtime & _gpiCxx_runtime;
       gaspi::Context & _gpiCxx_context;
       gaspi::segment::Segment & _gpiCxx_segment;

       // size of and a reference to the original vector
       int _origSize;
       const VarTYPE* _origVector;

       // treshold value used to compress the vector 
       VarTYPE _treshold;

       // flag indicating the "sign" ("yes" or "no") of the starting
       // run-length-sequence. The signs then alternatingly change.
       // "yes"-sequence <-> signumFlag = 1
       // "no"-sequence <-> signumFlag = 0       
       int _signumFlag;

       int _shrinkedSize; // vector size after the compression
       // compressed vector, with items/values greater than treshold,   
       // i.e. _shrinkedVector[i] >= treshold, i = 0..(_shrinkedSize-1).
       // This is the vector to be transferred, together with the
       // vector of run-lengthg codes, see below 
       std::unique_ptr<VarTYPE []> _shrinkedVector;
              
       int _runLengthSize; // size of the run-length vector
       // vector of run-lenth codes. The convention here:
       // This vector ALWAYS starts with the (number of) runs
       // for a meaningful values
       std::unique_ptr<int []>   _runLength;

       // working vectors
       std::vector<VarTYPE> _wrkVect;
       std::vector<int>     _wrkRunLength;

    public:
      CompressorRunLengths(  gaspi::Runtime & runTime
                           , gaspi::Context & context
	                   , gaspi::segment::Segment & segment
                          );

      ~CompressorRunLengths();

      void printCompressedVector(const char *fullPath) const;
      void printCompressedVector_inOriginalSize(const char *fullPath) const;
      void printRunLengthsVector(const char *fullPath) const;
      void printOriginalVector(const char *fullPath)  const;
 
      void compressVector(
            std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
          , int size                    // vector´s (original) size
	  , VarTYPE treshold);


  }; //CompressorRunLengths  


} //impl
} // namespace compressed_exchange

#endif // RUN_LENGTH_COMPRESS_H
