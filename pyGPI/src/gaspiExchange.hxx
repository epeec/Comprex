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

        bool first_send;
        bool first_recv;

        //void addGpiConnection_destination(gaspi::group::Rank destRank, int size, int tag){}

        //void addGpiConnection_source(){}

        virtual void communicateDataBufferSize_senderSide( int sizeBytes, gaspi::group::Rank destRank, int tag){
            //gaspi_printf("srcBuff\n");
            //if(!gaspi::isRuntimeAvailable()) gaspi_printf("Runtime not available!\n");

            //gaspi::singlesided::write::SourceBuffer srcBuff( _gpiCxx_segment, sizeof(int));
            if(first_send) {
                delete srcBuff_size;
                srcBuff_size = new gaspi::singlesided::write::SourceBuffer( _gpiCxx_segment, sizeof(int));
            }
            //gaspi_printf("connectTarget\n");
            srcBuff_size->connectToRemoteTarget(_gpiCxx_context, destRank, tag).waitForCompletion();

            int* buffEntry = (reinterpret_cast<int *>(srcBuff_size->address()));
            *buffEntry =  sizeBytes;

            //gaspi_printf("initTransfer\n");
            srcBuff_size->initTransfer(_gpiCxx_context);
        }

        virtual int communicateDataBufferSize_recverSide( gaspi::group::Rank srcRank, int tag){
            //gaspi::singlesided::write::TargetBuffer targBuff( _gpiCxx_segment, sizeof(int)) ;
            if(first_recv){
                delete targBuff_size;
                targBuff_size = new gaspi::singlesided::write::TargetBuffer( _gpiCxx_segment, sizeof(int));
            }
            targBuff_size->connectToRemoteSource(_gpiCxx_context, srcRank, tag).waitForCompletion();

            targBuff_size->waitForCompletion();

            int* buffEntry = (reinterpret_cast<int *>(targBuff_size->address()));
            return *buffEntry;
        }

        virtual void sendVectorToDestRank( const VarTYPE* vect, int size, gaspi::group::Rank destRank , int tag) {
            int sizeBytes = size * sizeof(VarTYPE);

            // gaspi_printf("Communicate Size (%d Bytes)\n",sizeBytes);
            // first communicate (send to dest rank) the buffer size
            communicateDataBufferSize_senderSide(sizeBytes, destRank, tag);

            // for(int i=0; i<size; ++i)
            //    gaspi_printf("vector[%d]=%#08x\n",i,vect[i]);

            // then communicate the data itself
            // alloc GaspiCxx source  buffer
            //gaspi_printf("srcBuffer\n");
            //gaspi::singlesided::write::SourceBuffer srcBuff(_gpiCxx_segment, sizeBytes);
            if(first_send){
                delete srcBuff_data;
                srcBuff_data = new gaspi::singlesided::write::SourceBuffer(_gpiCxx_segment, sizeBytes);
            }
            //gaspi_printf("connecting to target\n");
            srcBuff_data->connectToRemoteTarget(_gpiCxx_context, destRank, tag).waitForCompletion();
            writeBuffer(srcBuff_data->address(), vect, sizeBytes);

            //gaspi_printf("transfer\n");
            srcBuff_data->initTransfer(_gpiCxx_context);
            first_send=false;
        }

        virtual int getVectorFromSrcRank( VarTYPE* vect, int size, gaspi::group::Rank srcRank, int tag) {
            // first communicate (get from src rank) the buffer size
            int sizeBytes=communicateDataBufferSize_recverSide(srcRank, tag);
            if(size*sizeof(VarTYPE) < sizeBytes){
                char error_msg[100];
                sprintf(error_msg,"Received vector size (%d Bytes) is bigger than the buffer size (%d Bytes)!\n", sizeBytes, size*sizeof(VarTYPE));
                throw std::runtime_error(error_msg);
            }
            // then get  the data itself
            // alloc GaspiCxx  target- buffer
            //gaspi::singlesided::write::TargetBuffer targetBuff(_gpiCxx_segment, sizeBytes);
            if(first_recv){
                delete targBuff_data;
                targBuff_data = new gaspi::singlesided::write::TargetBuffer(_gpiCxx_segment, sizeBytes);
            }
            targBuff_data->connectToRemoteSource(_gpiCxx_context, srcRank, tag).waitForCompletion();
            targBuff_data->waitForCompletion();
            loadBuffer(vect, targBuff_data->address(), sizeBytes);
            first_recv=false;
            return sizeBytes/sizeof(VarTYPE);
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
                  first_send(true),
                  first_recv(true) {
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

    virtual void writeRemote( const VarTYPE* vector, int size, gaspi::group::Rank destRank, int tag) {
        // apply and update Rests with Threshold
        sendVectorToDestRank(vector, size, destRank, tag);
    }

    // returns the number of received data elements
    virtual int readRemote( std::vector<VarTYPE>& vector, gaspi::group::Rank srcRank, int tag) {
        return getVectorFromSrcRank(vector.data(), vector.size(), srcRank, tag);
    }

    // returns the number of received data elements
    virtual int readRemote( VarTYPE* vector, int size, gaspi::group::Rank srcRank, int tag) {
        return getVectorFromSrcRank(vector, size, srcRank, tag);
    }

}; // ComprEx

#endif //COMPRESSEDEXCHANGE_H