#ifndef COMPREX_H
#define COMPREX_H
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cstring>
#include <algorithm>

// GaspiCxx
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>

#include <GASPI_Ext.h>
// Comprex
#include <compressor.hxx>
#include <threshold.hxx>

#include <AbstractExchange.hxx>

/***************************************
 * ComprEx
 * Implements compressed point to point communication.
 ***************************************/
template <class VarTYPE>
class ComprEx : public AbstractExchange<VarTYPE> {
protected:
        // Compression type
        std::unique_ptr<Compressor<VarTYPE> > compressor;

        // Thresholding function
        std::unique_ptr<ThresholdFunction<VarTYPE> > threshold;

        // GaspiCxx
        gaspi::segment::Segment *gaspiCxx_segment;

        // Vector containing "rests" or residuals
        // In ComprEx, the error due to compression is accumulated locally in this vector.
        std::vector<VarTYPE> restsVect;

        // GaspiCXX communication endpoints
        gaspi::singlesided::write::SourceBuffer* srcBuff_data;
        gaspi::singlesided::write::TargetBuffer* targBuff_data;

        // source and target rank of this communication channel
        gaspi::group::Rank srcRank;
        gaspi::group::Rank targRank;

        // Size of the uncompressed vectors, in elements.
        // must be known in advance for the residual vector
        int size;
        // the size of the GaspiCXX communication buffers, in Bytes.
        // can be larger or smaller than `size`.
        // depends on the `size_factor`.
        int bufferSize;

        // Size factor for GaspiCXX send and receive buffers
        // as everything is compressed before it is send (also when flushing!), it is possible that the compressed vector is larger than the uncompressed one. Therefore, it is possible to allocate a multiple of the uncompressed vector size for the send/receive buffers.
        float size_factor;

        // flag which controls if buffersize in Bytes is also communicated.
        bool send_sizeBytes;

public:
    // Concstructor
    ComprEx();
    ComprEx(gaspi::segment::Segment* segment, int size, float size_factor=1.0, bool send_sizeBytes=false);

    // Destructor
    ~ComprEx();

    // Copy-Constructor
    ComprEx(const ComprEx<VarTYPE>& other);

    // Copy-assignment operator
    ComprEx<VarTYPE>& operator=(const ComprEx<VarTYPE>& other);

    void setCompressor(const Compressor<VarTYPE>& compressor);
    void setCompressor(const Compressor<VarTYPE>* compressor);

    void setThreshold(const ThresholdFunction<VarTYPE>& threshold);
    void setThreshold(const ThresholdFunction<VarTYPE>* threshold);

    // for debugging
    void print_rests();

    // Sets all residuals to zero.
    virtual void resetRests();

    // Send out the rests vector.
    // thresholding is not applied.
    virtual void flushRests();

    // Return a copy of the rests vector.
    std::vector<VarTYPE> getRests() const;

    // Copy the rests vector into another buffer.
    void getRests(VarTYPE *data, int size) const;

    // Add the rests vector to another buffer.
    void addRests(VarTYPE *data, int size) const;

    // Get a pointer to the buffer of the rests vector.
    const std::vector<VarTYPE>* getRests_p() const;

    // Write `vector` to remote target.
    // The target is specified in the setup phase. The data in `vector` is thresholded and it is added to the rests vector. 
    virtual void writeRemote( const VarTYPE* vector, int size);

    // Write `vector` to remote target.
    // only compression, but no thresholding or rests are applied.
    // the rests vector is unchanged.
    virtual void writeRemote_compressOnly( const VarTYPE* vector, int size);

    // Write `vector` to remote target.
    // compresses data with a threshold, but without residuals.
    // the rests vector is unchanged.
    void writeRemote_NoResidual( const VarTYPE* vector, int size);

    // Reads remote data.
    // the source is specified in the setup phase.
    // This operation overwrites all content in `vector`.
    virtual void readRemote( VarTYPE* vector, int size);

    // Read remote data.
    // fused operator, which reads target and adds result to `vector`, instead of overwriting them.
    virtual void readRemote_add(VarTYPE* vector, int size);

    // forward the compressed receive buffer and decompress it.
    void readRemote_forward(VarTYPE* vector, int size);

    // forward the compressed receive buffer and decompress it.
    void readRemote_forwardCopy(VarTYPE* vector, int size);

    void readRemote_wait();

    void forward_buffer();

    void forward_add(VarTYPE* vector, int size);

    virtual void connectTx(int targRank, int tag);

    // Sets up a connection to a target rank for sending.
    // the GaspiCXX data buffer is reinitialized.
    virtual void connectTx(gaspi::group::Rank targRank, int tag);

    virtual void connectRx(int srcRank, int tag);

    // Sets up a connection to a source rank for receiving.
    // the GaspiCXX data buffer is reinitialized.
    virtual void connectRx(gaspi::group::Rank srcRank, int tag);

    virtual void connectTo(int srcRank, int targRank, int tag);

    // A combination of `connectRx` and `connectTx`, simultaneously sets up send and receive connection.
    // usefull e.g. for ring connections, because waiting for completion only after both connections are issued.
    // srcRank: receive data from this rank
    // targRank: send data to this rank
    virtual void connectTo(gaspi::group::Rank srcRank, gaspi::group::Rank targRank, int tag);

}; // ComprEx


/***************************************
 * FastComprEx
 * Faster communication at the cost of staleness.
 ***************************************/
template <class VarTYPE>
class FastComprEx : public ComprEx<VarTYPE> {
protected:
    // size in bytes of buffer written in the previous doublebuffer phase
    int sizeBytes;
    // indicates which buffer should be used
    int doublebuffer_phase;
    // doublebuffer offset in bytes
    int doublebuffer_offset;
    // indicates the first exchange phase
    bool is_first;

    using ComprEx<VarTYPE>::restsVect;
    using ComprEx<VarTYPE>::srcBuff_data;

    // doublebuffer position, where data for sending stands.
    int get_doublebuffer_send_offset() const {
        return doublebuffer_offset * ((doublebuffer_phase+1)%2);
    }

    // doublebuffer position, where data should be written to.
    int get_doublebuffer_write_offset() const {
        return doublebuffer_offset * doublebuffer_phase;
    }

    // swap doublebuffer positions.
    void update_doublebuffer_phase(){
        doublebuffer_phase = (doublebuffer_phase+1)%2;
    }

public:
    FastComprEx();

    FastComprEx(gaspi::segment::Segment* segment, int size, float size_factor=1.0);

    FastComprEx<VarTYPE>(const FastComprEx<VarTYPE>& other);

    FastComprEx<VarTYPE>& operator=(const FastComprEx<VarTYPE>& other);

    virtual void flushRests();

    virtual void resetRests();

    virtual void writeRemote( const VarTYPE* vector, int size);

    virtual void writeRemote_compressOnly(VarTYPE* vector, int size);

    // Sends the compressed buffer from another FastComprEx communicator.
    virtual void writeRemote_sameAs( const FastComprEx* other);

    virtual void connectTx_sameBufferAs(int targRank, int tag, const FastComprEx<VarTYPE>* other);
    // Use the source buffer of another FastComprEx to connect to another rank.
    virtual void connectTx_sameBufferAs(gaspi::group::Rank targRank, int tag, const FastComprEx<VarTYPE>* other);

};

#endif // #define COMPREX_H
