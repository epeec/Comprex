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

        void sendCompressedVectorToDestRank( const CompressedVector<VarTYPE>* cVect ) {
            if(!srcBuff_data){
                throw std::runtime_error("Send issued without connecting to Remote!");
            }
            int sizeBytes = cVect->calcSizeBytes();

            cVect->writeBuffer( srcBuff_data->address() );
            srcBuff_data->initTransferPart(_gpiCxx_context, sizeBytes, 0);
        }

        void getCompressedVectorFromSrcRank(std::unique_ptr<CompressedVector<VarTYPE> >* output) {
            if(!targBuff_data){
                throw std::runtime_error("Receive issued without connecting to Remote!");
            }

            targBuff_data->waitForCompletion();
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
                  srcBuff_data(NULL),
                  targBuff_data(NULL),
                  size(size){
        resetRests();
    }

    ~ComprEx() {
        if(srcBuff_data) delete srcBuff_data;
        if(targBuff_data) delete targBuff_data;
    }

    void setCompressor(const Compressor<VarTYPE>& compressor) {
        this->compressor = compressor.copy();
    }

    void setCompressor(const Compressor<VarTYPE>* compressor) {
        this->compressor = compressor->copy();
    }

    void setThreshold(const ThresholdFunction<VarTYPE>& threshold) {
        this->threshold = threshold.copy();
    }

    void setThreshold(const ThresholdFunction<VarTYPE>* threshold) {
        this->threshold = threshold->copy();
    }

    void resetRests(){
        for(int i=0; i<restsVect.size(); ++i){
            restsVect[i]=0;
        }
        //std::memset(static_cast<void *> (_restsVector.data()), 0, _origSize*sizeof(VarTYPE));
    }

    void flushRests(){
        // TODO: don't compress Rests vector when flushing it!
        // Workaround: don't apply thresholding
        compressor->compress(&cVect, &restsVect);
        sendCompressedVectorToDestRank(cVect.get());
        resetRests();
    }

    std::vector<VarTYPE> getRests() const{
        return restsVect;
    }

    const std::vector<VarTYPE>* getRests_p() const{
        return &restsVect;
    }

    void writeRemote( const std::vector<VarTYPE>* vector) {
        // apply and update Rests with Threshold
        applyRests(&txVect, vector);
        // send Data
        compressor->compress(&cVect, &txVect);
        sendCompressedVectorToDestRank(cVect.get());
    }

    void writeRemote( const VarTYPE* vector, int size) {
        // apply and update Rests with Threshold
        applyRests(&txVect, vector, size);
        // send Data
        compressor->compress(&cVect, &txVect);
        sendCompressedVectorToDestRank(cVect.get());
    }

    void readRemote( std::vector<VarTYPE>* vector) {
        getCompressedVectorFromSrcRank(&cVect);
        compressor->decompress(vector, cVect.get());
    }

    void readRemote( VarTYPE* vector, int size) {
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

        srcBuff_data = new gaspi::singlesided::write::SourceBuffer(_gpiCxx_segment, this->size*sizeof(VarTYPE)*size_factor);
        auto handle_sbdata(srcBuff_data->connectToRemoteTarget(_gpiCxx_context, targRank, tag));
        
        targBuff_data = new gaspi::singlesided::write::TargetBuffer(_gpiCxx_segment, this->size*sizeof(VarTYPE)*size_factor);
        auto handle_tbdata(targBuff_data->connectToRemoteSource(_gpiCxx_context, srcRank, tag));

        handle_sbdata.waitForCompletion();
        handle_tbdata.waitForCompletion();
    }

}; // ComprEx

#endif //COMPRESSEDEXCHANGE_H