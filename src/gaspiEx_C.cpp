#include "gaspiEx.hxx"
#include "gaspiEx.tpp"
#include "gaspiEnvironment.hxx"
#include <stdio.h>
#include <vector>
#include <GASPI_Ext.h>

extern "C"
{

typedef float data_t;

///////////////////////
// GaspiEx
//////////////////////
GaspiEx<data_t>* GaspiEx_new(GaspiEnvironment* gaspi_env, int size) { 
    return new GaspiEx<data_t>(gaspi_env->get_segment(), size);
}
void GaspiEx_del(GaspiEx<data_t>* self) { delete self;}
void GaspiEx_writeRemote(GaspiEx<data_t>* self, const data_t* vector, int size) {
    //std::vector<data_t> vec(vector, vector+size);
    self->writeRemote( vector, size );
}
void GaspiEx_readRemote(GaspiEx<data_t>* self, data_t* vector, int size) {
    self->readRemote( vector, size );
}

void GaspiEx_connectTo(GaspiEx<data_t>* self, int srcRank, int targRank, int tag) {
    self->connectTo( static_cast<gaspi::group::Rank>(srcRank), static_cast<gaspi::group::Rank>(targRank), tag );
}

void GaspiEx_connectTx(GaspiEx<data_t>* self, int targRank, int tag){
    self->connectTx( static_cast<gaspi::group::Rank>(targRank), tag );
}

void GaspiEx_connectRx(GaspiEx<data_t>* self, int srcRank, int tag){
    self->connectRx( static_cast<gaspi::group::Rank>(srcRank), tag );
}

} // extern "C"
