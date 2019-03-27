// runLengthCompressr.hxx -> run-length-encoding compression, definitions

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring> // memcpy

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

       // GaspiCxx source- and target buffers 
       //std::unique_ptr< gaspi::singlesided::write::SourceBuffer > _sourceBuff;
       //std::unique_ptr< gaspi::singlesided::write::TargetBuffer > _targetBuff;

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
              
       int _runLengthSize; // size of the run-length vector

       // work-vectors, where also the final result is stored.
       // 
       // The compressed vector, with items/values greater than treshold,   
       // i.e. _shrinkedVector[i] >= treshold, i = 0..(_shrinkedSize-1).
       // This is the vector to be transferred, together with the
       // vector of run-lengthg codes, see below 
       std::vector<VarTYPE> _wrkVect;

       // vector of run-lenth codes. 
       std::vector<int>     _wrkRunLength;


       // GaspiCxx transfer
       std::size_t _buffSizeBytes;

       void compressVectorSingleThreaded(
            std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
          , int size                    // vector´s (original) size
	  , VarTYPE treshold);      // treshold       

       
       void communicateDataBufferSize_senderSide(gaspi::group::Rank  destRank
                                                 , int tag);  
       void communicateDataBufferSize_recverSide(gaspi::group::Rank  srcRank
					         , int tag);

       void calculateSizeSourceBuffer();
       void fillInSourceBuffer(gaspi::singlesided::write::SourceBuffer &srcBuff);

       void deCompressVector_inLocalStructs(
		       gaspi::singlesided::write::TargetBuffer &targBuff);
                          //   , std::unique_ptr<VarTYPE []> & vector);
     
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
 
      const VarTYPE* getPtrShrinkedVector() const;
      const int* getPtrRunLengthsVector() const;

      const int getSizeCompressedVector() const;
      const int getSizeRunLengthsVector() const;
      const int getSignumFlag() const;
      const VarTYPE getTreshold() const;
       

      void sendCompressedVectorToDestRank(
		         gaspi::group::Rank  destRank // destination rank
			 , int tag);

      void getCompressedVectorFromSrcRank(
                            std::unique_ptr<VarTYPE []>  & vector
			  , gaspi::group::Rank  srcRank // source rank
			  , int tag);
     
      void compress_and_p2pVectorWriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to src vector
               , int size                     // vector´s (original) size
	       , VarTYPE treshold             // treshold
	       , gaspi::group::Rank  destRank // destination rank
               , int tag                     // message tag
               , int nThreads = 1);          // number of threads used in compression

      void p2pVectorGetRemote(
	       std::unique_ptr<VarTYPE []>  & vector // pointer to dest vector
               , int size                    // vector´s (original) size
	       , gaspi::group::Rank  srcRank// source rank
               , int tag                    // message tag
               , int nThreads = 1);         // number of threads used in de-compression


  }; //CompressorRunLengths  

} //impl
} // namespace compressed_exchange

#include <runLenCompressr.cxx>

#endif // RUN_LENGTH_COMPRESS_H
