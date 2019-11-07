#include "gaspiExchange.hxx"
#include <stdio.h>
#include <vector>
#include <GASPI_Ext.h>

extern "C"
{

typedef float data_t;

///////////////////////
// ComprEx
//////////////////////
GaspiEx<data_t>* GaspiEx_new( gaspi::Runtime* runTime, gaspi::Context* context, gaspi::segment::Segment* segment) { 
    return new GaspiEx<data_t>((*runTime), (*context), (*segment));
}
void GaspiEx_del(GaspiEx<data_t>* self) { delete self;}
void GaspiEx_writeRemote(GaspiEx<data_t>* self, const data_t* vector, int size, int destRank, int tag) {
    //std::vector<data_t> vec(vector, vector+size);
    self->writeRemote( vector, size, static_cast<gaspi::group::Rank>(destRank), tag);
}
int GaspiEx_readRemote(GaspiEx<data_t>* self, data_t* vector, int size, int srcRank, int tag) {
    return self->readRemote( vector, size, static_cast<gaspi::group::Rank>(srcRank), tag);
}

} // extern "C"
