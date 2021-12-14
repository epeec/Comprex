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
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/group/Rank.hpp>
//#include <GaspiCxx/segment/Allocator.hpp>
//#include <GaspiCxx/segment/NotificationManager.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
//#include <GaspiCxx/utility/ScopedAllocation.hpp>

#include <GASPI_Ext.h>

#include "gaspiEx.hxx"

/***************************************
 * GaspiEx
 * Similar like ComprEx, but without any compression scheme.
 * A simple point to point communication.
 ***************************************/
template <class VarTYPE>
GaspiEx<VarTYPE>::GaspiEx()
            : gaspiCxx_segment(NULL),
              srcBuff_data(NULL),
              targBuff_data(NULL),
              srcRank(-1),
              targRank(-1),
              size(0) {}

template <class VarTYPE>
GaspiEx<VarTYPE>::GaspiEx(gaspi::segment::Segment* segment, int size)
            : gaspiCxx_segment(segment),
                srcBuff_data(NULL),
                targBuff_data(NULL),
                srcRank(-1),
                targRank(-1),
                size(size) {}

template <class VarTYPE>
GaspiEx<VarTYPE>::GaspiEx(const GaspiEx<VarTYPE>& other)
        : GaspiEx() {
    *this = other;
}

template <class VarTYPE>
GaspiEx<VarTYPE>::~GaspiEx() {
    if(srcBuff_data) delete srcBuff_data;
    if(targBuff_data) delete targBuff_data;
}

template <class VarTYPE>
GaspiEx<VarTYPE>& GaspiEx<VarTYPE>::operator=(const GaspiEx<VarTYPE>& other){
    if(srcBuff_data != NULL or targBuff_data != NULL){
        throw std::runtime_error("Communicator with established connection cannot be assigned to other object!");
    }
    gaspiCxx_segment = other.gaspiCxx_segment;
    size = other.size;
    return *this;
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::writeRemote( const VarTYPE* vector, int size) {
    int sizeBytes = size * sizeof(VarTYPE);
    std::memcpy( srcBuff_data->address(), vector, sizeBytes );
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes, 0);
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::writeRemote( int size) {
    int sizeBytes = size * sizeof(VarTYPE);
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes, 0);
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::readRemote( VarTYPE* vector, int size) {
    targBuff_data->waitForCompletion();
    int sizeBytes = size * sizeof(VarTYPE);
    std::memcpy(vector, targBuff_data->address(), sizeBytes);
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::readRemote_add( VarTYPE* vector, int size) {
    targBuff_data->waitForCompletion();
    int sizeBytes = size * sizeof(VarTYPE);
    VarTYPE* pBuffer = static_cast<VarTYPE*>(targBuff_data->address());
    for(int i=0; i<size; ++i){
        vector[i] += pBuffer[i];
    }
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::readRemote_wait(){
    targBuff_data->waitForCompletion();
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::forward_buffer(){
    int sizeBytes = size * sizeof(VarTYPE);
    std::memcpy(srcBuff_data->address(), targBuff_data->address(), sizeBytes);
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::forward_add(VarTYPE* vector, int size){
    int sizeBytes = size * sizeof(VarTYPE);
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes, 0);
    VarTYPE* pBuffer = static_cast<VarTYPE*>(srcBuff_data->address());
    for(int i=0; i<size; ++i){
        vector[i] += pBuffer[i];
    }
}


template <class VarTYPE>
void GaspiEx<VarTYPE>::connectTx(int targRank, int tag){
    connectTx(gaspi::group::Rank(targRank), tag);
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::connectTx(gaspi::group::Rank targRank, int tag){
    this->targRank = targRank;
    if(srcBuff_data) delete srcBuff_data;

    srcBuff_data = new gaspi::singlesided::write::SourceBuffer(*gaspiCxx_segment, size*sizeof(VarTYPE));
    auto handle_sbdata(srcBuff_data->connectToRemoteTarget(gaspi::group::Group(), targRank, tag));
    handle_sbdata.waitForCompletion();
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::connectTx_sameBufferAs(int targRank, int tag, const GaspiEx<VarTYPE>* other){
    connectTx_sameBufferAs(gaspi::group::Rank(targRank), tag, other);
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::connectTx_sameBufferAs(gaspi::group::Rank targRank, int tag, const GaspiEx<VarTYPE>* other){
    if(!other->srcBuff_data){
        throw std::runtime_error("Cannot connected with buffer of unconnected communicator!");
    }
    this->targRank = targRank;
    if(srcBuff_data) delete srcBuff_data;

    srcBuff_data = new gaspi::singlesided::write::SourceBuffer(other->srcBuff_data->address(), *gaspiCxx_segment, size*sizeof(VarTYPE));
    auto handle_sbdata(srcBuff_data->connectToRemoteTarget(gaspi::group::Group(), targRank, tag));
    handle_sbdata.waitForCompletion();
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::connectRx(int srcRank, int tag){
    connectRx(gaspi::group::Rank(srcRank), tag);
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::connectRx(gaspi::group::Rank srcRank, int tag){
    this->srcRank = srcRank;
    if(targBuff_data) delete targBuff_data;

    targBuff_data = new gaspi::singlesided::write::TargetBuffer(*gaspiCxx_segment, size*sizeof(VarTYPE));
    auto handle_tbdata(targBuff_data->connectToRemoteSource(gaspi::group::Group(), srcRank, tag));
    handle_tbdata.waitForCompletion();
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::connectTo(int srcRank, int targRank, int tag){
    connectTo(gaspi::group::Rank(srcRank), gaspi::group::Rank(targRank), tag);
}

template <class VarTYPE>
void GaspiEx<VarTYPE>::connectTo(gaspi::group::Rank srcRank, gaspi::group::Rank targRank, int tag) {
    // set new ranks
    this->srcRank = srcRank;
    this->targRank = targRank;
    // cleanup old connection
    if(srcBuff_data) delete srcBuff_data;
    if(targBuff_data) delete targBuff_data;

    srcBuff_data = new gaspi::singlesided::write::SourceBuffer(*gaspiCxx_segment, size*sizeof(VarTYPE));
    auto handle_sbdata(srcBuff_data->connectToRemoteTarget(gaspi::group::Group(), targRank, tag));
    
    targBuff_data = new gaspi::singlesided::write::TargetBuffer(*gaspiCxx_segment, size*sizeof(VarTYPE));
    auto handle_tbdata(targBuff_data->connectToRemoteSource(gaspi::group::Group(), srcRank, tag));

    handle_sbdata.waitForCompletion();
    handle_tbdata.waitForCompletion();
}
