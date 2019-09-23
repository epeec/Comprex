/*
 *  comprex.hxx
 *
 *
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cstring>
#include <algorithm>


#include  <GaspiCxx/GaspiCxx.hpp>

#include <mThrRLEcompress.hxx>
#include <mThrTopKcompress.hxx>

#ifndef COMPREX_H
#define COMPREX_H

namespace compressed_exchange {

  // compression type
  enum class CompressionType {
      runLengthEncoding,     // run-length-encoding (RLE) 
      sparseIndexing         // sparse indexing   (SI)
  };


  template <class VarTYPE>
  class ComprEx {
       
    protected:
      
       // compression type
       CompressionType  _type;

       // GaspiCxx 
       gaspi::Runtime & _gpiCxx_runtime;
       gaspi::Context & _gpiCxx_context;
       gaspi::segment::Segment & _gpiCxx_segment;
           
       // size of and a reference to the original vector
       int _origSize;
       const VarTYPE* _origVector;

       // vector containing "rests" : the left-over after the communication
       std::unique_ptr< VarTYPE [] > _restsVector;

       // treshold value used to compress the vector 
       VarTYPE _treshold;

        // vector size after the compression
       int _shrinkedSize; 
              
        // size of the auxiliary-info vector
       int _auxInfoVectSize;

       // work-vectors, where also the final result is stored.
       // 
       // The compressed vector, with items/values greater than treshold,   
       // i.e. _shrinkedVector[i] >= treshold, i = 0..(_shrinkedSize-1).
       // This is the vector to be transferred, together with the
       // auxiliary-info-vector, e.g. of run-lengthg codes, see below 
       std::vector<VarTYPE> _shrinkedVect; //_wrkVect

       // auxiliary  info vector, i.e. containing auxiliary information, 
       // in addition to the compressed vector values. This means
       // - vector of run-lenth codes, for run-length-encoding
       // - vector containing the  indices of the non-zeros, in sparse-indexing 
       std::vector<int>     _auxInfoVectr; 

       // GaspiCxx transfer
       std::size_t _buffSizeBytes;


       // pin_pattern, number of threads, in case of a multi-threaded compression
       int _nThreads;
       const int* _pinPattern;
       std::unique_ptr< pthread_t [] >  _pThreads;

       // check if (_origVector[i] + _restsVector[i]) > _treshold;
       virtual bool checkFulfilled_forItem(int i);

       virtual void compressVectorSingleThreaded(
            std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
	    , VarTYPE treshold) = 0;      // treshold       

       
       virtual void communicateDataBufferSize_senderSide(
                                               gaspi::group::Rank  destRank
                                             , int tag);  
       virtual void communicateDataBufferSize_recverSide(
                                               gaspi::group::Rank  srcRank
					     , int tag);

       virtual void calculateSizeSourceBuffer() = 0;
       virtual void fillInSourceBuffer(
                       gaspi::singlesided::write::SourceBuffer &srcBuff) = 0;

       virtual void deCompressVector_inLocalStructs(
		       gaspi::singlesided::write::TargetBuffer &targBuff) = 0;
     
       virtual void fillInVectorFromLocalStructs(
                                     std::unique_ptr<VarTYPE []> & vector) = 0;

      virtual void sendCompressedVectorToDestRank(
		         gaspi::group::Rank  destRank // destination rank
			 , int tag);

      virtual void getCompressedVectorFromSrcRank(
                            std::unique_ptr<VarTYPE []>  & vector
			  , gaspi::group::Rank  srcRank // source rank
			  , int tag);


       virtual void erazeTheLocalStructures();   // _auxInfoVect, _shrinkedVect, ..
        

    public:
      ComprEx( 
                gaspi::Runtime & runTime
              , gaspi::Context & context
	      , gaspi::segment::Segment & segment
	      , int origSize   // size of the vectors to work with
	      , int nThreads = 1
	      , const int* _pinPattern = NULL
	       );

      ~ComprEx();

      virtual int rank() const {
          return static_cast<int>(_gpiCxx_context.rank().get());}

      virtual void flushTheRestsVector();
      virtual const VarTYPE * entryPointerRestsVector() const;

      virtual void printCompressedVector(const char *fullPath) const;
      virtual void printCompressedVector_inOriginalSize(
                                               const char *fullPath) const = 0;
 
      virtual void printAuxiliaryInfoVector(const char *fullPath) const = 0;
      virtual void printOriginalVector(const char *fullPath)  const;
 
      virtual const VarTYPE* getPtrShrinkedVector() const;
      virtual const int* getPtrAuxiliaryInfoVector() const;

      virtual const int getSizeCompressedVector() const;
      virtual const int getSizeAuxiliaryInfoVector() const;
     
      virtual  const VarTYPE getTreshold() const;
            
      virtual void setPinPatternForTheThreads(
		       //int nThtreads,       // number of threads per rank
                       std::unique_ptr<int []> const & pinPattern); // pin pattern

      virtual void compress_and_p2pVectorWriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to src vector
	       , VarTYPE treshold             // treshold
	       , gaspi::group::Rank  destRank // destination rank
               , int tag                     // message tag
               , int nThreads = 1);          // number of threads used in compression

      virtual void p2pVectorGetRemote(
	       std::unique_ptr<VarTYPE []>  & vector // pointer to dest vector
	       , gaspi::group::Rank  srcRank// source rank
               , int tag                    // message tag
               , int nThreads = 1);         // number of threads used in de-compression

  }; // ComprEx 
  

  template <class VarTYPE>
  class ComprExRunLengths: public ComprEx<VarTYPE> {

    private:

       // flag indicating the "sign" ("yes" or "no") of the starting
       // run-length-sequence. The signs then alternatingly change.
       // "yes"-sequence <-> signumFlag = 1
       // "no"-sequence <-> signumFlag = 0       
       int _signumFlag;


       std::unique_ptr< MultiThreadedRLE<VarTYPE> > _mThrLRE;

       void compressVectorSingleThreaded(
            std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
	  , VarTYPE treshold);      // treshold       

       void calculateSizeSourceBuffer();
       void fillInSourceBuffer(gaspi::singlesided::write::SourceBuffer &srcBuff);

       void deCompressVector_inLocalStructs(
		       gaspi::singlesided::write::TargetBuffer &targBuff);
     
       void fillInVectorFromLocalStructs(std::unique_ptr<VarTYPE []> & vector);

    public:
      ComprExRunLengths(  gaspi::Runtime & runTime
                           , gaspi::Context & context
	                   , gaspi::segment::Segment & segment
	                   , int origSize   // size of the vectors to work with
                          );

      ~ComprExRunLengths();

      void printCompressedVector_inOriginalSize(const char *fullPath) const;
      void printAuxiliaryInfoVector(const char *fullPath) const;

      const int getSignumFlag() const;

      void compress_and_p2pVectorWriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
	       , VarTYPE treshold            // treshold
	       , gaspi::group::Rank  destRank// destination rank
               , int tag                     // message tag
               , int nThreads = 1);   // number of threads used in compression     

  }; // class ComprExRunLengths 


  template <class VarTYPE>
  class ComprExTopK: public ComprEx<VarTYPE> {

    private:
 
      // counter of the current p2p-send-call after intantiating the class
      int _crrWriteCall;

      std::unique_ptr< MultiThreadedTopK<VarTYPE> > _mThrTopK;

      // work-vector
      std::vector<PairIndexValue<VarTYPE> > _vectPairs;

      static bool sortPredicate_absValue(const PairIndexValue<VarTYPE> & obj1, 
			                 const PairIndexValue<VarTYPE> & obj2)
      {
	return std::abs(obj1.val) > std::abs(obj2.val);
      }

    
      void calculateNumberOfItemsToSend(VarTYPE topK_percents);
      void fillIn_absValuesVector_and_sort(); 
      void calculateSizeSourceBuffer();

      void compressVectorSingleThreaded(
            std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
	  , VarTYPE topK_percents);      // the top K-percents,  the value of K
     
      void fillInSourceBuffer(
		       gaspi::singlesided::write::SourceBuffer &srcBuff);

      void inRestsVectorSetToZeroTheItemsJustSent();

      void deCompressVector_inLocalStructs(
			gaspi::singlesided::write::TargetBuffer &targBuff);
     
      void fillInVectorFromLocalStructs(
			 std::unique_ptr<VarTYPE []> & vector);

    public:

      ComprExTopK( gaspi::Runtime & runTime
                   , gaspi::Context & context
	           , gaspi::segment::Segment & segment
	           , int origSize   // size of the vectors to work with
                   , int nThreads = 0
	           , const int* pinPattern = NULL
                 );

      ~ComprExTopK();
   
      void printCompressedVector_inOriginalSize(const char *fullPath) const {};

      void printAuxiliaryInfoVector(const char *fullPath) const; // base-class function
    //void printIndexValuePairsVector(int itr, const char *fullPath) const;

      const struct PairIndexValue<VarTYPE> *
                      entryPointerVectorPairs()  const;
      
      void compress_and_p2pVectorWriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
	       , VarTYPE topK_percents        // top K-percents of the items to be transferred
	       , gaspi::group::Rank  destRank// destination rank
               , int tag);                     // message tag      

 }; // class ComprExTopK


  template <class VarTYPE>
  class ComprExSparseIdx : public ComprEx<VarTYPE> {
    // ...

  };//class ComprExSparseIdx 



}  // end namespace compressed_exchange

// include the template implementation
#include <comprex.cxx>
#include <runLenComprEx.cxx>
#include <mThrRLEcompress.cxx>
#include <topKcomprEx.cxx>
#include <mThrTopKcompress.cxx>

#endif // #define COMPREX_H
