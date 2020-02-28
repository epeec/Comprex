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

/***************************************
 * ComprEx
 ***************************************/
template <class VarTYPE>
class ComprEx {
protected:
        // compression type
        std::unique_ptr<Compressor<VarTYPE> > compressor;

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

        // communication endpoints
        gaspi::singlesided::write::SourceBuffer* srcBuff_data;
        gaspi::singlesided::write::TargetBuffer* targBuff_data;

        // source and target of this communication channel
        gaspi::group::Rank srcRank;
        gaspi::group::Rank targRank;

        // size of the uncompressed vectors
        // must be known for residual vector
        int size;

        // internal buffer size (NOT the size of the communicated data!)

        // virtual void communicateDataBufferSize_senderSide( int sizeBytes, gaspi::group::Rank destRank, int tag){
        //     //gaspi_printf("srcBuff\n");
        //     //if(!gaspi::isRuntimeAvailable()) gaspi_printf("Runtime not available!\n");

        //     //gaspi::segment::Segment gpiCxx_segment(1000);
        //     gaspi::singlesided::write::SourceBuffer srcBuff( _gpiCxx_segment, sizeof(int)) ;

        //     //gaspi_printf("connectTarget\n");
        //     srcBuff.connectToRemoteTarget(_gpiCxx_context, destRank, tag).waitForCompletion();

        //     int* buffEntry = (reinterpret_cast<int*>(srcBuff.address()));
        //     *buffEntry =  sizeBytes;

        //     //gaspi_printf("initTransfer\n");
        //     srcBuff.initTransfer(_gpiCxx_context);
        // }

        // virtual int communicateDataBufferSize_recverSide( gaspi::group::Rank srcRank, int tag){
        //     gaspi::singlesided::write::TargetBuffer targBuff( _gpiCxx_segment, sizeof(int)) ;

        //     targBuff.connectToRemoteSource(_gpiCxx_context, srcRank, tag).waitForCompletion();

        //     targBuff.waitForCompletion();

        //     int* buffEntry = (reinterpret_cast<int *>(targBuff.address()));
        //     return *buffEntry;
        // }

        virtual void sendCompressedVectorToDestRank( const CompressedVector<VarTYPE>* cVect ) {
            if(!srcBuff_data){
                throw std::runtime_error("Send issued without connecting to Remote!");
            }
            int sizeBytes = cVect->calcSizeBytes();
            // // copy size of compressed vector at the beginning of the buffer
            // std::memcpy(srcBuff_data->address(), &sizeBytes, sizeof(int));
            // write compressed vector after the total size
            // cVect->writeBuffer(srcBuff_data->address()+sizeof(int));

            cVect->writeBuffer( srcBuff_data->address() );
            srcBuff_data->initTransferPart(_gpiCxx_context, sizeBytes, 0);

            //gaspi_printf("transfer\n");
            // only transfer a part of the buffer
            // int totalSizeBytes = sizeBytes + sizeof(int);
            // srcBuff_data->initTransferPart(_gpiCxx_context, totalSizeBytes, 0);
        }

        virtual void getCompressedVectorFromSrcRank(std::unique_ptr<CompressedVector<VarTYPE> >* output) {
            if(!targBuff_data){
                throw std::runtime_error("Receive issued without connecting to Remote!");
            }

            targBuff_data->waitForCompletion();

            // get total size of compressed vector:
            // int sizeBytes;
            // std::memcpy(&sizeBytes, targBuff_data->address(), sizeof(int) );
            // output->get()->loadBuffer
            *output = compressor->getEmptyCompressedVector();
            output->get()->loadBuffer(targBuff_data->address());
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
                  restsVect(std::vector<VarTYPE>(size)),
                  srcRank(-1),
                  targRank(-1),
                  size(size){
        resetRests();
        // srcBuff_data = new  gaspi::singlesided::write::SourceBuffer( _gpiCxx_segment, 0);
        // targBuff_data= new  gaspi::singlesided::write::TargetBuffer( _gpiCxx_segment, 0);
        srcBuff_data = NULL;
        targBuff_data= NULL;
    }

    ~ComprEx() {
        if(srcBuff_data) delete srcBuff_data;
        if(targBuff_data) delete targBuff_data;
    }

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
    virtual void flushRests(){
        // send restsVect without applying thresholding
        compressor->compress(&cVect, &restsVect);
        sendCompressedVectorToDestRank(cVect.get());
        resetRests();
    }
    virtual std::vector<VarTYPE> getRests() const{
        return restsVect;
    }

    virtual const std::vector<VarTYPE>* getRests_p() const{
        return &restsVect;
    }

    virtual void writeRemote( const std::vector<VarTYPE>* vector) {
        // apply and update Rests with Threshold
        applyRests(&txVect, vector);
        // send Data
        compressor->compress(&cVect, &txVect);
        sendCompressedVectorToDestRank(cVect.get());
    }

    virtual void writeRemote( const VarTYPE* vector, int size) {
        // apply and update Rests with Threshold
        applyRests(&txVect, vector, size);
        // send Data
        compressor->compress(&cVect, &txVect);
        sendCompressedVectorToDestRank(cVect.get());
    }

    virtual void readRemote( std::vector<VarTYPE>* vector) {
        getCompressedVectorFromSrcRank(&cVect);
        compressor->decompress(vector, cVect.get());
    }

    virtual void readRemote( VarTYPE* vector, int size) {
        getCompressedVectorFromSrcRank(&cVect);
        compressor->decompress(vector, size, cVect.get());
    }


    void connectTx(gaspi::group::Rank targRank, int tag, int size_factor=1){
        this->targRank = targRank;
        if(srcBuff_data) delete srcBuff_data;

        srcBuff_data = new gaspi::singlesided::write::SourceBuffer(_gpiCxx_segment, this->size*sizeof(VarTYPE)*size_factor);
        auto handle_sbdata(srcBuff_data->connectToRemoteTarget(_gpiCxx_context, targRank, tag));
        handle_sbdata.waitForCompletion();
    }

    void connectRx(gaspi::group::Rank srcRank, int tag, int size_factor=1){
        this->srcRank = srcRank;
        if(targBuff_data) delete targBuff_data;

        targBuff_data = new gaspi::singlesided::write::TargetBuffer(_gpiCxx_segment, this->size*sizeof(VarTYPE)*size_factor);
        auto handle_tbdata(targBuff_data->connectToRemoteSource(_gpiCxx_context, srcRank, tag));
        handle_tbdata.waitForCompletion();
    }

    // srcRank: receive data from this rank
    // targRank: send data to this rank
    void connectTo(gaspi::group::Rank srcRank, gaspi::group::Rank targRank, int tag, int size_factor=1){
        // set new ranks
        this->srcRank = srcRank;
        this->targRank = targRank;

        // cleanup old connection
        if(srcBuff_data) delete srcBuff_data;
        if(targBuff_data) delete targBuff_data;

        // gaspi_printf("Data Connection");
        srcBuff_data = new gaspi::singlesided::write::SourceBuffer(_gpiCxx_segment, this->size*sizeof(VarTYPE)*size_factor);
        auto handle_sbdata(srcBuff_data->connectToRemoteTarget(_gpiCxx_context, targRank, tag));
        
        targBuff_data = new gaspi::singlesided::write::TargetBuffer(_gpiCxx_segment, this->size*sizeof(VarTYPE)*size_factor);
        auto handle_tbdata(targBuff_data->connectToRemoteSource(_gpiCxx_context, srcRank, tag));

        handle_sbdata.waitForCompletion();
        handle_tbdata.waitForCompletion();
    }

}; // ComprEx

#endif //COMPRESSEDEXCHANGE_H