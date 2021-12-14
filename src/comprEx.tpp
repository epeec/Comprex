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
//#include <GaspiCxx/segment/NotificationManager.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
//#include <GaspiCxx/utility/ScopedAllocation.hpp>

#include <GASPI_Ext.h>
// Comprex
#include <compressor.hxx>
#include <threshold.hxx>

#include <AbstractExchange.hxx>
#include <comprEx.hxx>

/***************************************
 * ComprEx
 * Implements compressed point to point communication.
 ***************************************/
template <class VarTYPE>
ComprEx<VarTYPE>::ComprEx() 
            : compressor(std::unique_ptr<Compressor<VarTYPE> >{new CompressorNone<VarTYPE>()}),
                threshold(std::unique_ptr<ThresholdFunction<VarTYPE> >{new ThresholdNone<VarTYPE>()}),
                gaspiCxx_segment(NULL),
                restsVect(std::vector<VarTYPE>(0)),
                srcRank(-1),
                targRank(-1),
                size(0),
                bufferSize(0),
                size_factor(0),
                send_sizeBytes(false){
    srcBuff_data = NULL;
    targBuff_data= NULL;
}

template <class VarTYPE>
ComprEx<VarTYPE>::ComprEx(gaspi::segment::Segment* segment, int size, float size_factor, bool send_sizeBytes)
            : compressor(std::unique_ptr<Compressor<VarTYPE> >{new CompressorNone<VarTYPE>()}),
                threshold(std::unique_ptr<ThresholdFunction<VarTYPE> >{new ThresholdNone<VarTYPE>()}),
                gaspiCxx_segment(segment),
                restsVect(std::vector<VarTYPE>(0)), // only allocate memory for rests vector, if there is an outgoing connection!
                srcRank(-1),
                targRank(-1),
                size(size),
                size_factor(size_factor),
                send_sizeBytes(send_sizeBytes){
    srcBuff_data = NULL;
    targBuff_data= NULL;
    bufferSize = (int)(size*sizeof(VarTYPE)*size_factor);
}

template <class VarTYPE>
ComprEx<VarTYPE>::~ComprEx() {
    if(srcBuff_data) delete srcBuff_data;
    if(targBuff_data) delete targBuff_data;
}

template <class VarTYPE>
ComprEx<VarTYPE>::ComprEx(const ComprEx<VarTYPE>& other)
        : ComprEx() {
    *this = other;
}


template <class VarTYPE>
ComprEx<VarTYPE>& ComprEx<VarTYPE>::operator=(const ComprEx<VarTYPE>& other){
    if(srcBuff_data != NULL or targBuff_data != NULL){
        throw std::runtime_error("ComprEx object with established connection cannot be assigned to other object!");
    }
    gaspiCxx_segment = other.gaspiCxx_segment;
    compressor.reset(other.compressor->clone());
    threshold.reset(other.threshold->clone());
    restsVect.resize(other.restsVect.size());
    this->resetRests();
    size = other.size;
    bufferSize = other.bufferSize;
    size_factor=other.size_factor;
    send_sizeBytes=other.send_sizeBytes;
    return *this;
}

template <class VarTYPE>
void ComprEx<VarTYPE>::setCompressor(const Compressor<VarTYPE>& compressor) {
    this->compressor.reset(compressor.clone());
}

template <class VarTYPE>
void ComprEx<VarTYPE>::setCompressor(const Compressor<VarTYPE>* compressor) {
    this->compressor.reset(compressor->clone());
}

template <class VarTYPE>
void ComprEx<VarTYPE>::setThreshold(const ThresholdFunction<VarTYPE>& threshold) {
    this->threshold.reset(threshold.clone());
}

template <class VarTYPE>
void ComprEx<VarTYPE>::setThreshold(const ThresholdFunction<VarTYPE>* threshold) {
    this->threshold.reset(threshold->clone());
}

template <class VarTYPE>
void ComprEx<VarTYPE>::print_rests(){
    for(int it=0; it<restsVect.size(); ++it){
        gaspi_printf("Rests[%d]: %d\n", it, restsVect[it]);
    }
}

template <class VarTYPE>
void ComprEx<VarTYPE>::resetRests(){
    for(int i=0; i<restsVect.size(); ++i){
        restsVect[i]=0;
    }
}

template <class VarTYPE>
void ComprEx<VarTYPE>::flushRests(){
    if(!srcBuff_data){
        throw std::runtime_error("Send issued without connecting to Remote!");
    }
    // compress data and get size of compressed buffer in bytes
    int offset = (send_sizeBytes) ? sizeof(int) : 0;
    int sizeBytes = compressor->compress(srcBuff_data->address()+offset, restsVect.data(), restsVect.size(), 0);
    if(send_sizeBytes){
        *(reinterpret_cast<int*>(srcBuff_data->address())) = sizeBytes;
    }
    // initialize non-blocking send
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes+offset, 0);
    resetRests();
}

template <class VarTYPE>
std::vector<VarTYPE> ComprEx<VarTYPE>::getRests() const{
    return restsVect;
}

template <class VarTYPE>
void ComprEx<VarTYPE>::getRests(VarTYPE *data, int size) const{
    if(size != restsVect.size()){
        char error_msg[100];
        sprintf(error_msg,"Vector of size %d does not match Rests Vector of size %d!\n", size, (int)restsVect.size());
        throw std::runtime_error(error_msg);
    }
    memcpy(data, restsVect.data(), sizeof(VarTYPE)*size );
}

template <class VarTYPE>
void ComprEx<VarTYPE>::addRests(VarTYPE *data, int size) const{
    if(size != restsVect.size()){
        char error_msg[100];
        sprintf(error_msg,"Vector of size %d does not match Rests Vector of size %d!\n", size, (int)restsVect.size());
        throw std::runtime_error(error_msg);
    }
    for(int it=0; it<size; ++it){
        data[it] += restsVect[it];
    }
}

template <class VarTYPE>
const std::vector<VarTYPE>* ComprEx<VarTYPE>::getRests_p() const{
    return &restsVect;
}

template <class VarTYPE>
void ComprEx<VarTYPE>::writeRemote( const VarTYPE* vector, int size) {
    // check if a connection is already established
    if(!srcBuff_data){
        throw std::runtime_error("Send issued without connecting to Remote!");
    }
    // calculate new threshold value
    threshold->prepare_with(restsVect.data(), restsVect.size(), vector, size);
    int offset = (send_sizeBytes) ? sizeof(int) : 0;
    int sizeBytes = compressor->compress_add_update(srcBuff_data->address()+offset, restsVect.data(), restsVect.size(), vector, size, threshold->get_threshold());
    if(send_sizeBytes){
        *(reinterpret_cast<int*>(srcBuff_data->address())) = sizeBytes;
    }
    // initialize non-blocking send
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes+offset, 0);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::writeRemote_compressOnly( const VarTYPE* vector, int size){
    // check if a connection is already established
    if(!this->srcBuff_data){
        throw std::runtime_error("Send issued without connecting to Remote!");
    }
    // compress, but without using the thresholding (assuming, that data is already sparse).
    int offset = (send_sizeBytes) ? sizeof(int) : 0;
    int sizeBytes = this->compressor->compress(srcBuff_data->address()+offset, vector, size, 0);
    if(send_sizeBytes){
        *(reinterpret_cast<int*>(srcBuff_data->address())) = sizeBytes;
    }
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes+offset, 0);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::writeRemote_NoResidual( const VarTYPE* vector, int size){
    // check if a connection is already established
    if(!this->srcBuff_data){
        throw std::runtime_error("Send issued without connecting to Remote!");
    }
    threshold->prepare(vector, size);
    // compress
    int offset = (send_sizeBytes) ? sizeof(int) : 0;
    int sizeBytes = this->compressor->compress(srcBuff_data->address()+offset, vector, size, threshold->get_threshold());
    if(send_sizeBytes){
        *(reinterpret_cast<int*>(srcBuff_data->address())) = sizeBytes;
    }
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes+offset, 0);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::readRemote( VarTYPE* vector, int size) {
    // check if connection is established
    if(!targBuff_data){
            throw std::runtime_error("Receive issued without connecting to Remote!");
    }
    // wait until data is received
    targBuff_data->waitForCompletion();
    int offset = (send_sizeBytes) ? sizeof(int) : 0;
    int rx_vector_size = compressor->decompress(vector, size, targBuff_data->address()+offset);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::readRemote_add(VarTYPE* vector, int size) {
    // check if connection is established
    if(!targBuff_data){
            throw std::runtime_error("Receive issued without connecting to Remote!");
    }
    // wait until data is received
    targBuff_data->waitForCompletion();
    int offset = (send_sizeBytes) ? sizeof(int) : 0;
    compressor->add(vector, size, targBuff_data->address()+offset);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::readRemote_forward(VarTYPE* vector, int size){
    targBuff_data->waitForCompletion();
    if(send_sizeBytes){
        int offset = sizeof(int);
        int sizeBytes = *(reinterpret_cast<int*>(targBuff_data->address()));
        std::memcpy(srcBuff_data->address(), targBuff_data->address(), sizeBytes+offset);
        srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes+offset, 0);
        compressor->decompress(vector, size, targBuff_data->address()+offset);
    }
    else {
        int sizeBytes = compressor->decompress(vector, size, targBuff_data->address());
        std::memcpy(srcBuff_data->address(), targBuff_data->address(), sizeBytes);
        srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes, 0);
        
    }
}

template <class VarTYPE>
void ComprEx<VarTYPE>::readRemote_forwardCopy(VarTYPE* vector, int size){
    if(!this->send_sizeBytes){
        throw std::runtime_error("If using readRemote_forwardCopy, send_sizeBytes must be true!");
    }
    targBuff_data->waitForCompletion();
    int offset = sizeof(int);
    int sizeBytes = *(reinterpret_cast<int*>(targBuff_data->address()));
    std::memcpy(srcBuff_data->address(), targBuff_data->address(), sizeBytes+offset);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::readRemote_wait(){
    targBuff_data->waitForCompletion();
}

template <class VarTYPE>
void ComprEx<VarTYPE>::forward_buffer(){
    if(!this->send_sizeBytes){
        throw std::runtime_error("If using forward_buffer, send_sizeBytes must be true!");
    }
    int offset = sizeof(int);
    int sizeBytes = *(reinterpret_cast<int*>(targBuff_data->address()));
    std::memcpy(srcBuff_data->address(), targBuff_data->address(), sizeBytes+offset);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::forward_add(VarTYPE* vector, int size){
    if(!this->send_sizeBytes){
        throw std::runtime_error("If using forward_add, send_sizeBytes must be true!");
    }
    int offset = sizeof(int);
    int sizeBytes = *(reinterpret_cast<int*>(srcBuff_data->address()));
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes+offset, 0);
    compressor->add(vector, size, srcBuff_data->address()+offset);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::connectTx(int targRank, int tag){
    connectTx(gaspi::group::Rank(targRank), tag);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::connectTx(gaspi::group::Rank targRank, int tag){
    this->targRank = targRank;
    if(srcBuff_data) delete srcBuff_data;
    // allocate memory for restsVector
    restsVect.resize(size);
    resetRests();

    srcBuff_data = new gaspi::singlesided::write::SourceBuffer(*gaspiCxx_segment, bufferSize);
    auto handle_sbdata(srcBuff_data->connectToRemoteTarget(gaspi::group::Group(), targRank, tag));
    handle_sbdata.waitForCompletion();
}

template <class VarTYPE>
void ComprEx<VarTYPE>::connectRx(int srcRank, int tag){
    connectRx(gaspi::group::Rank(srcRank), tag);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::connectRx(gaspi::group::Rank srcRank, int tag){
    this->srcRank = srcRank;
    if(targBuff_data) delete targBuff_data;

    targBuff_data = new gaspi::singlesided::write::TargetBuffer(*gaspiCxx_segment, bufferSize);
    auto handle_tbdata(targBuff_data->connectToRemoteSource(gaspi::group::Group(), srcRank, tag));
    handle_tbdata.waitForCompletion();
}

template <class VarTYPE>
void ComprEx<VarTYPE>::connectTo(int srcRank, int targRank, int tag){
    connectTo(gaspi::group::Rank(srcRank), gaspi::group::Rank(targRank), tag);
}

template <class VarTYPE>
void ComprEx<VarTYPE>::connectTo(gaspi::group::Rank srcRank, gaspi::group::Rank targRank, int tag){
    // set new ranks
    this->srcRank = srcRank;
    this->targRank = targRank;

    // cleanup old connection
    if(srcBuff_data) delete srcBuff_data;
    if(targBuff_data) delete targBuff_data;

    // only allocate memory for restsVector, if there is an outgoing connection
    restsVect.resize(size);
    resetRests();

    srcBuff_data = new gaspi::singlesided::write::SourceBuffer(*gaspiCxx_segment, bufferSize);
    auto handle_sbdata(srcBuff_data->connectToRemoteTarget(gaspi::group::Group(), targRank, tag));
    
    targBuff_data = new gaspi::singlesided::write::TargetBuffer(*gaspiCxx_segment, bufferSize);
    auto handle_tbdata(targBuff_data->connectToRemoteSource(gaspi::group::Group(), srcRank, tag));

    handle_sbdata.waitForCompletion();
    handle_tbdata.waitForCompletion();
}


/***************************************
 * FastComprEx
 * Faster communication at the cost of staleness.
 ***************************************/

template <class VarTYPE>
FastComprEx<VarTYPE>::FastComprEx() 
        : ComprEx<VarTYPE>::ComprEx()
        , sizeBytes(0)
        , doublebuffer_phase(0)
        , is_first(true) 
        {
}

template <class VarTYPE>
FastComprEx<VarTYPE>::FastComprEx(gaspi::segment::Segment* segment, int size, float size_factor)
        // double size_factor for double buffering
        : ComprEx<VarTYPE>::ComprEx(segment, size, size_factor*2)
        , sizeBytes(0)
        , doublebuffer_phase(0)
        , is_first(true) 
        {
    doublebuffer_offset = (int)(size * size_factor * sizeof(VarTYPE));
}

template <class VarTYPE>
FastComprEx<VarTYPE>::FastComprEx(const FastComprEx<VarTYPE>& other) {
    *this = other;    
}

template <class VarTYPE>
FastComprEx<VarTYPE>& FastComprEx<VarTYPE>::operator=(const FastComprEx<VarTYPE>& other){
    ComprEx<VarTYPE>::operator=(other);
    doublebuffer_offset = other.doublebuffer_offset;
    is_first=true;
    return *this;
}

template <class VarTYPE>
void FastComprEx<VarTYPE>::flushRests(){
    // send restsVect without applying thresholding
    if(!srcBuff_data){
        throw std::runtime_error("Send issued without connecting to Remote!");
    }
    // the data to be send stands at this offset. Intercept it and then send it out.
    int send_offset = get_doublebuffer_send_offset();
    // decompress and add source buffer to rests vector
    this->compressor->add(restsVect.data(), restsVect.size(), srcBuff_data->address()+send_offset);
    // compress data and get size of compressed buffer in bytes
    sizeBytes = this->compressor->compress(srcBuff_data->address()+send_offset, restsVect.data(), restsVect.size(), 0);
    // initialize non-blocking send
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes, send_offset);
    ComprEx<VarTYPE>::resetRests();
    is_first=true;
}

template <class VarTYPE>
void FastComprEx<VarTYPE>::resetRests(){
    ComprEx<VarTYPE>::resetRests();
    is_first=true;
}

template <class VarTYPE>
void FastComprEx<VarTYPE>::writeRemote( const VarTYPE* vector, int size) {
    // check if a connection is already established
    if(!this->srcBuff_data){
        throw std::runtime_error("Send issued without connecting to Remote!");
    }
    int write_offset = get_doublebuffer_write_offset();
    int send_offset =  get_doublebuffer_send_offset();
    if(is_first){
        // restsVect should be all zeros at this point, so sizeBytes~0.
        sizeBytes = this->compressor->compress_update(this->srcBuff_data->address()+send_offset, restsVect.data(), restsVect.size(), this->threshold->get_threshold());
        is_first=false;
    }
    // initialize non-blocking send with data from previous iteration
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes, send_offset);
    // prepare threshold, but temporarily add rests and vector elements
    this->threshold->prepare_with(restsVect.data(), restsVect.size(), vector, size);
    // compress, but also add rests and vector
    sizeBytes = this->compressor->compress_add_update(srcBuff_data->address()+write_offset, restsVect.data(), restsVect.size(), vector, size, this->threshold->get_threshold());
    update_doublebuffer_phase();
}

template <class VarTYPE>
void FastComprEx<VarTYPE>::writeRemote_compressOnly(VarTYPE* vector, int size){
    // check if a connection is already established
    if(!this->srcBuff_data){
        throw std::runtime_error("Send issued without connecting to Remote!");
    }
    int write_offset = get_doublebuffer_write_offset();
    int send_offset =  get_doublebuffer_send_offset();
    if(is_first){
        // restsVect should be all zeros at this point, so sizeBytes~0.
        sizeBytes = this->compressor->compress(this->srcBuff_data->address()+send_offset, restsVect.data(), restsVect.size(), 0);
        is_first=false;
    }
    // initialize non-blocking send with data from previous iteration
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes, send_offset);
    // compress
    sizeBytes = this->compressor->compress(srcBuff_data->address()+write_offset, vector, size, 0);
    update_doublebuffer_phase();
}

template <class VarTYPE>
void FastComprEx<VarTYPE>::writeRemote_sameAs( const FastComprEx* other) {
    // check if a connection is already established
    if(!this->srcBuff_data){
        throw std::runtime_error("Send issued without connecting to Remote!");
    }
    int write_offset = get_doublebuffer_write_offset();
    int send_offset =  get_doublebuffer_send_offset();
    if(is_first){
        // restsVect should be all zeros at this point.
        sizeBytes = this->compressor->compress_update(this->srcBuff_data->address()+send_offset, restsVect.data(), restsVect.size(), this->threshold->get_threshold());
        is_first=false;
    }
    // initialize non-blocking send with data from previous iteration
    srcBuff_data->initTransferPart(gaspi::getRuntime(), sizeBytes, send_offset);
    // copy data from other ComprEx
    if(!other->is_first){
        std::memcpy(this->srcBuff_data->address()+write_offset, other->srcBuff_data->address()+other->get_doublebuffer_send_offset(), other->sizeBytes);
        this->sizeBytes = other->sizeBytes;
    } else {
        this->is_first=true;
    }
    update_doublebuffer_phase();
}

template <class VarTYPE>
void FastComprEx<VarTYPE>::connectTx_sameBufferAs(int targRank, int tag, const FastComprEx<VarTYPE>* other){
    connectTx_sameBufferAs(gaspi::group::Rank(targRank), tag, other);
}

template <class VarTYPE>
void FastComprEx<VarTYPE>::connectTx_sameBufferAs(gaspi::group::Rank targRank, int tag, const FastComprEx<VarTYPE>* other){
    if(!other->srcBuff_data){
        throw std::runtime_error("Cannot connected with buffer of unconnected communicator!");
    }
    this->targRank = targRank;
    if(srcBuff_data) delete srcBuff_data;
    // only create restsVector, if there is an outgoing connection
    restsVect.resize(this->size);
    ComprEx<VarTYPE>::resetRests();

    srcBuff_data = new gaspi::singlesided::write::SourceBuffer(other->srcBuff_data->address(), *(this->gaspiCxx_segment), (int)(this->size*sizeof(VarTYPE)*this->size_factor));
    auto handle_sbdata(srcBuff_data->connectToRemoteTarget(gaspi::group::Group(), targRank, tag));
    handle_sbdata.waitForCompletion();
}
