#ifndef GASPIEX_H
#define GASPIEX_H

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

#include "AbstractExchange.hxx"

/***************************************
 * GaspiEx
 * Similar like ComprEx, but without any compression scheme.
 * A simple point to point communication.
 ***************************************/
template <class VarTYPE>
class GaspiEx : public AbstractExchange<VarTYPE> {
protected:
        // GaspiCxx
        gaspi::segment::Segment* gaspiCxx_segment;

        gaspi::singlesided::write::SourceBuffer* srcBuff_data;
        gaspi::singlesided::write::TargetBuffer* targBuff_data;

        gaspi::group::Rank srcRank;
        gaspi::group::Rank targRank;

        // size of the communicated vector. Needs to be known in advance, because of singlesided communication.
        int size;

public:

    GaspiEx();

    GaspiEx(gaspi::segment::Segment* segment, int size);

    GaspiEx(const GaspiEx<VarTYPE>& other);

    ~GaspiEx();

    GaspiEx<VarTYPE>& operator=(const GaspiEx<VarTYPE>& other);

    virtual void writeRemote( const VarTYPE* vector, int size);

    virtual void writeRemote( int size);

    virtual void readRemote( VarTYPE* vector, int size);

    virtual void readRemote_add( VarTYPE* vector, int size);

    void readRemote_wait();

    void forward_buffer();

    void forward_add(VarTYPE* vector, int size);


    virtual void connectTx(int targRank, int tag);

    virtual void connectTx(gaspi::group::Rank targRank, int tag);

    virtual void connectTx_sameBufferAs(int targRank, int tag, const GaspiEx<VarTYPE>* other);

    virtual void connectTx_sameBufferAs(gaspi::group::Rank targRank, int tag, const GaspiEx<VarTYPE>* other);

    virtual void connectRx(int srcRank, int tag);

    virtual void connectRx(gaspi::group::Rank srcRank, int tag);

    virtual void connectTo(int srcRank, int targRank, int tag);

    // srcRank: receive data from that rank
    // targRank: send data to that rank
    virtual void connectTo(gaspi::group::Rank srcRank, gaspi::group::Rank targRank, int tag);

}; // ComprEx

#endif //GASPIEX_H
