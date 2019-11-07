#ifndef COMPRESSEDEXCHANGE_H
#define COMPRESSEDEXCHANGE_H

#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cstring>
#include <algorithm>

//#include  <GaspiCxx/GaspiCxx.hpp>
/*
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
#include <GaspiCxx/utility/ScopedAllocation.hpp>
*/

// GaspiCxx
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Allocator.hpp>
#include <GaspiCxx/segment/NotificationManager.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
#include <GaspiCxx/utility/ScopedAllocation.hpp>

#include <GASPI_Ext.h>
// Comprex
#include <compressor.hxx>
#include <threshold.hxx>

// compression type
enum class CompressionType {
    runLengthEncoding,     // run-length-encoding (RLE)
    sparseIndexing         // sparse indexing   (SI)
};

/***************************************
 * ComprEx
 ***************************************/
template <class VarTYPE>
class ComprEx {
protected:
        // compression type
        std::unique_ptr<Compressor<VarTYPE> > compressor;

        // no compression
        CompressorNone<VarTYPE> compressor_none;

        // thresholding function to increase sparsity
        std::unique_ptr<ThresholdFunction<VarTYPE> > threshold;

        // storing intermediate values
        std::vector<VarTYPE> txVect;
        std::unique_ptr<CompressedVector<VarTYPE> > cVect;

        // GaspiCxx
        gaspi::Runtime& _gpiCxx_runtime;
        gaspi::Context& _gpiCxx_context;
        gaspi::segment::Segment& _gpiCxx_segment;

        // vector containing "rests" : the left-over after the communication
        std::vector<VarTYPE> restsVect;

        virtual void communicateDataBufferSize_senderSide( int sizeBytes, gaspi::group::Rank destRank, int tag){
            //gaspi_printf("srcBuff\n");
            //if(!gaspi::isRuntimeAvailable()) gaspi_printf("Runtime not available!\n");

            //gaspi::segment::Segment gpiCxx_segment(1000);
            gaspi::singlesided::write::SourceBuffer srcBuff( _gpiCxx_segment, sizeof(int)) ;

            //gaspi_printf("connectTarget\n");
            srcBuff.connectToRemoteTarget(_gpiCxx_context, destRank, tag).waitForCompletion();

            int* buffEntry = (reinterpret_cast<int*>(srcBuff.address()));
            *buffEntry =  sizeBytes;

            //gaspi_printf("initTransfer\n");
            srcBuff.initTransfer(_gpiCxx_context);
        }

        virtual int communicateDataBufferSize_recverSide( gaspi::group::Rank srcRank, int tag){
            gaspi::singlesided::write::TargetBuffer targBuff( _gpiCxx_segment, sizeof(int)) ;

            targBuff.connectToRemoteSource(_gpiCxx_context, srcRank, tag).waitForCompletion();

            targBuff.waitForCompletion();

            int* buffEntry = (reinterpret_cast<int *>(targBuff.address()));
            return *buffEntry;
        }

        virtual void sendCompressedVectorToDestRank( const CompressedVector<VarTYPE>* cVect, gaspi::group::Rank destRank , int tag) {
            int sizeBytes = cVect->calcSizeBytes();

            //gaspi_printf("Communicate Size\n");
            // first communicate (send to dest rank) the buffer size
            communicateDataBufferSize_senderSide(sizeBytes, destRank, tag);

            // then communicate the data itself
            // alloc GaspiCxx source  buffer
            //gaspi_printf("srcBuffer\n");
            gaspi::singlesided::write::SourceBuffer srcBuff(_gpiCxx_segment, sizeBytes) ;

            //gaspi_printf("connecting to target\n");
            srcBuff.connectToRemoteTarget(_gpiCxx_context, destRank, tag).waitForCompletion();
            cVect->writeBuffer(srcBuff.address());

            //gaspi_printf("transfer\n");
            srcBuff.initTransfer(_gpiCxx_context);
        }

        virtual void getCompressedVectorFromSrcRank(std::unique_ptr<CompressedVector<VarTYPE> >* output, gaspi::group::Rank srcRank, int tag) {
            // first communicate (get from src rank) the buffer size
            int sizeBytes=communicateDataBufferSize_recverSide(srcRank, tag);

            // then get  the data itself
            // alloc GaspiCxx  target- buffer
            gaspi::singlesided::write::TargetBuffer targetBuff(_gpiCxx_segment, sizeBytes);

            targetBuff.connectToRemoteSource(_gpiCxx_context, srcRank, tag).waitForCompletion();
            targetBuff.waitForCompletion();
            *output = compressor->getEmptyCompressedVector();
            output->get()->loadBuffer(targetBuff.address());
        }

        void applyRests(std::vector<VarTYPE>* output, const std::vector<VarTYPE>* vector){
            // check if input vector and rests are of same size
            if(vector->size() != restsVect.size()){
                char error_msg[100];
                sprintf(error_msg,"Input Vector of size %d does not match Rests Vector of size %d!\n", (int)vector->size(), (int)restsVect.size());
                throw std::runtime_error(error_msg);
            }
            // add Input Vector to Rests
            for(int i=0; i<restsVect.size(); ++i){
                restsVect[i] += (*vector)[i];
            }
            // apply Threshold
            threshold->cut(output, &restsVect);
            // updated Rests
            for(int i=0; i<restsVect.size(); ++i){
                restsVect[i] -= (*output)[i];
            }
        }

        void applyRests(std::vector<VarTYPE>* output, const VarTYPE* vector, int size){
            // check if input vector and rests are of same size
            if(size != restsVect.size()){
                char error_msg[100];
                sprintf(error_msg,"Input Vector of size %d does not match Rests Vector of size %d!\n", (int)size, (int)restsVect.size());
                throw std::runtime_error(error_msg);
            }
            // add Input Vector to Rests
            for(int i=0; i<restsVect.size(); ++i){
                restsVect[i] += vector[i];
            }
            // apply Threshold
            threshold->cut(output, &restsVect);
            // updated Rests
            for(int i=0; i<restsVect.size(); ++i){
                restsVect[i] -= (*output)[i];
            }
        }

public:

    ComprEx(gaspi::Runtime & runTime, gaspi::Context & context, gaspi::segment::Segment & segment, int size)
                : compressor(std::unique_ptr<Compressor<VarTYPE> >{new CompressorNone<VarTYPE>()}),
                  threshold(std::unique_ptr<ThresholdFunction<VarTYPE> >{new ThresholdNone<VarTYPE>()}),
                  _gpiCxx_runtime(runTime),
                  _gpiCxx_context(context),
                  _gpiCxx_segment(segment),
                  restsVect(std::vector<VarTYPE>(size)) {
        resetRests();
    }

    ~ComprEx() {}

    virtual void setCompressor(const Compressor<VarTYPE>& compressor) {
        this->compressor = compressor.copy();
    }
    virtual void setCompressor(const Compressor<VarTYPE>* compressor) {
        this->compressor = compressor->copy();
    }

    virtual void setThreshold(const ThresholdFunction<VarTYPE>& threshold) {
        this->threshold = threshold.copy();
    }
    virtual void setThreshold(const ThresholdFunction<VarTYPE>* threshold) {
        this->threshold = threshold->copy();
    }

    virtual void resetRests(){
        for(int i=0; i<restsVect.size(); ++i){
            restsVect[i]=0;
        }
        //std::memset(static_cast<void *> (_restsVector.data()), 0, _origSize*sizeof(VarTYPE));
    }
    virtual void flushRests(gaspi::group::Rank destRank, int tag){
        // send restsVect without applying thresholding
        compressor->compress(&cVect, &restsVect);
        sendCompressedVectorToDestRank(cVect.get(), destRank, tag);
        resetRests();
    }
    virtual std::vector<VarTYPE> getRests() const{
        return restsVect;
    }

    virtual const std::vector<VarTYPE>* getRests_p() const{
        return &restsVect;
    }

    virtual void writeRemote( const std::vector<VarTYPE>* vector, gaspi::group::Rank destRank, int tag) {
        // apply and update Rests with Threshold
        applyRests(&txVect, vector);
        // send Data
        compressor->compress(&cVect, &txVect);
        sendCompressedVectorToDestRank(cVect.get(), destRank, tag);
    }

    virtual void writeRemote( const VarTYPE* vector, int size, gaspi::group::Rank destRank, int tag) {
        // apply and update Rests with Threshold
        applyRests(&txVect, vector, size);
        // send Data
        compressor->compress(&cVect, &txVect);
        sendCompressedVectorToDestRank(cVect.get(), destRank, tag);
    }

    virtual void readRemote( std::vector<VarTYPE>* vector, gaspi::group::Rank srcRank, int tag) {
        getCompressedVectorFromSrcRank(&cVect, srcRank, tag);
        compressor->decompress(vector, cVect.get());
    }

    virtual void readRemote( VarTYPE* vector, int size, gaspi::group::Rank srcRank, int tag) {
        getCompressedVectorFromSrcRank(&cVect, srcRank, tag);
        /*
        int cVect_size = cVect.get()->uncompressed_size();
        if(size != cVect_size){
            char error_msg[100];
            sprintf(error_msg,"Received Vector of size %d does not match the buffer of size %d!\n", cVect_size, (int)size);
            throw std::runtime_error(error_msg);
        }
        */
        compressor->decompress(vector, size, cVect.get());
    }

}; // ComprEx

#endif //COMPRESSEDEXCHANGE_H