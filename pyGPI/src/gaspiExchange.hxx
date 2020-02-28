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

/***************************************
 * GaspiEx
 ***************************************/
template <class VarTYPE>
class GaspiEx {
protected:
        // GaspiCxx
        gaspi::Runtime& _gpiCxx_runtime;
        gaspi::Context& _gpiCxx_context;
        gaspi::segment::Segment& _gpiCxx_segment;

        gaspi::singlesided::write::SourceBuffer* srcBuff_size;
        gaspi::singlesided::write::SourceBuffer* srcBuff_data;

        gaspi::singlesided::write::TargetBuffer* targBuff_size;
        gaspi::singlesided::write::TargetBuffer* targBuff_data;

        gaspi::group::Rank srcRank;
        gaspi::group::Rank targRank;

        int size;

        virtual void communicateDataBufferSize_senderSide( int sizeBytes, gaspi::group::Rank destRank, int tag){

            int* buffEntry = (reinterpret_cast<int *>(srcBuff_size->address()));
            *buffEntry =  sizeBytes;

            // gaspi_printf("initTransfer\n");
            srcBuff_size->initTransfer(_gpiCxx_context);
        }

        virtual int communicateDataBufferSize_recverSide( gaspi::group::Rank srcRank, int tag){
            
            // gaspi_printf("Getting Data buffer size\n");
            targBuff_size->waitForCompletion();
            int* buffEntry = (reinterpret_cast<int *>(targBuff_size->address()));
            // gaspi_printf("Data buffer size: %d\n",*buffEntry);
            return *buffEntry;
        }

        virtual void sendVectorToDestRank( const VarTYPE* vect, int size, gaspi::group::Rank destRank , int tag) {
            int sizeBytes = size * sizeof(VarTYPE);

            //gaspi_printf("connecting to target\n");
            writeBuffer(srcBuff_data->address(), vect, sizeBytes);

            //gaspi_printf("transfer\n");
            srcBuff_data->initTransfer(_gpiCxx_context);
        }

        virtual int getVectorFromSrcRank( VarTYPE* vect, int size, gaspi::group::Rank srcRank, int tag) {
            targBuff_data->waitForCompletion();
            loadBuffer(vect, targBuff_data->address(), size*sizeof(VarTYPE));
            return size;
        }

        void writeBuffer(void* dest_buffer, const VarTYPE* src_buffer, int sizeBytes) {
            // write data
            std::memcpy( dest_buffer, src_buffer, sizeBytes );
        }

        virtual void loadBuffer(VarTYPE* dest_buffer, const void* src_buffer, int sizeBytes) {
            // load data into vector
            std::memcpy(dest_buffer, src_buffer, sizeBytes );
        }

public:

    GaspiEx(gaspi::Runtime & runTime, gaspi::Context & context, gaspi::segment::Segment & segment)
                : _gpiCxx_runtime(runTime),
                  _gpiCxx_context(context),
                  _gpiCxx_segment(segment),
                  srcRank(-1),
                  targRank(-1),
                  size(0) {
        srcBuff_data = new  gaspi::singlesided::write::SourceBuffer( _gpiCxx_segment, 0);
        srcBuff_size = new  gaspi::singlesided::write::SourceBuffer( _gpiCxx_segment, 0);
        targBuff_data= new  gaspi::singlesided::write::TargetBuffer( _gpiCxx_segment, 0);
        targBuff_size= new  gaspi::singlesided::write::TargetBuffer( _gpiCxx_segment, 0);
    }

    ~GaspiEx() {
        delete srcBuff_data;
        delete srcBuff_size;
        delete targBuff_data;
        delete targBuff_size;
    }

    virtual void writeRemote( const std::vector<VarTYPE>& vector, gaspi::group::Rank destRank, int tag) {
        // send Data
        sendVectorToDestRank( vector.data(), vector.size(), destRank, tag);
    }

    virtual void writeRemote( const std::vector<VarTYPE>* vector, gaspi::group::Rank destRank, int tag) {
        // send Data
        sendVectorToDestRank( vector->data(), vector->size(), destRank, tag);
    }

    virtual void writeRemote( const VarTYPE* vector, int size, gaspi::group::Rank destRank, int tag) {
        // apply and update Rests with Threshold
        sendVectorToDestRank(vector, size, destRank, tag);
    }

    // returns the number of received data elements
    virtual int readRemote( std::vector<VarTYPE>& vector, gaspi::group::Rank srcRank, int tag) {
        return getVectorFromSrcRank(vector.data(), vector.size(), srcRank, tag);
    }

    virtual int readRemote( std::vector<VarTYPE>* vector, gaspi::group::Rank srcRank, int tag) {
        return getVectorFromSrcRank(vector->data(), vector->size(), srcRank, tag);
    }

    // returns the number of received data elements
    virtual int readRemote( VarTYPE* vector, int size, gaspi::group::Rank srcRank, int tag) {
        return getVectorFromSrcRank(vector, size, srcRank, tag);
    }

    void connectTx(gaspi::group::Rank targRank, int size, int tag){
        this->targRank = targRank;
        this->size = size;
        delete srcBuff_data;
        delete srcBuff_size;

        srcBuff_size = new gaspi::singlesided::write::SourceBuffer( _gpiCxx_segment, sizeof(int));
        auto handle_sbsize(srcBuff_size->connectToRemoteTarget(_gpiCxx_context, targRank, tag));
        handle_sbsize.waitForCompletion();

        srcBuff_data = new gaspi::singlesided::write::SourceBuffer(_gpiCxx_segment, size*sizeof(VarTYPE));
        auto handle_sbdata(srcBuff_data->connectToRemoteTarget(_gpiCxx_context, targRank, tag));
        handle_sbdata.waitForCompletion();
    }

    void connectRx(gaspi::group::Rank srcRank, int size, int tag){
        this->srcRank = srcRank;
        this->size = size;
        delete targBuff_data;
        delete targBuff_size;

        targBuff_size = new gaspi::singlesided::write::TargetBuffer( _gpiCxx_segment, sizeof(int));
        auto handle_tbsize( targBuff_size->connectToRemoteSource(_gpiCxx_context, srcRank, tag) );
        handle_tbsize.waitForCompletion();

        targBuff_data = new gaspi::singlesided::write::TargetBuffer(_gpiCxx_segment, size*sizeof(VarTYPE));
        auto handle_tbdata(targBuff_data->connectToRemoteSource(_gpiCxx_context, srcRank, tag));
        handle_tbdata.waitForCompletion();
    }

    // srcRank: receive data from this rank
    // targRank: send data to this rank
    void connectTo(gaspi::group::Rank srcRank, gaspi::group::Rank targRank, int size, int tag){
        // set new ranks
        this->srcRank = srcRank;
        this->targRank = targRank;
        this->size = size;

        // cleanup old connection
        delete srcBuff_data;
        delete srcBuff_size;
        delete targBuff_data;
        delete targBuff_size;
    
        // gaspi_printf("Sizes Connection\n");
        srcBuff_size = new gaspi::singlesided::write::SourceBuffer( _gpiCxx_segment, sizeof(int));
        auto handle_sbsize(srcBuff_size->connectToRemoteTarget(_gpiCxx_context, targRank, tag));
        
        targBuff_size = new gaspi::singlesided::write::TargetBuffer( _gpiCxx_segment, sizeof(int));
        auto handle_tbsize( targBuff_size->connectToRemoteSource(_gpiCxx_context, srcRank, tag) );

        handle_sbsize.waitForCompletion();
        handle_tbsize.waitForCompletion();

        // gaspi_printf("Data Connection");
        srcBuff_data = new gaspi::singlesided::write::SourceBuffer(_gpiCxx_segment, size*sizeof(VarTYPE));
        auto handle_sbdata(srcBuff_data->connectToRemoteTarget(_gpiCxx_context, targRank, tag));
        
        targBuff_data = new gaspi::singlesided::write::TargetBuffer(_gpiCxx_segment, size*sizeof(VarTYPE));
        auto handle_tbdata(targBuff_data->connectToRemoteSource(_gpiCxx_context, srcRank, tag));

        handle_sbdata.waitForCompletion();
        handle_tbdata.waitForCompletion();
    }

}; // ComprEx

#endif //COMPRESSEDEXCHANGE_H