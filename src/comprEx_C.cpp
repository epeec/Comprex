#include "comprEx.hxx"
#include "comprEx.tpp"
#include "compressor.hxx"
#include "threshold.hxx"
#include "gaspiEnvironment.hxx"
#include <stdio.h>
#include <vector>

extern "C"
{

typedef float data_t;

///////////////////////
// ComprEx
//////////////////////
ComprEx<data_t>* Comprex_new(GaspiEnvironment* gaspi_env, int size, float size_factor) { 
    return new ComprEx<data_t>(gaspi_env->get_segment(), size, size_factor);
}
void Comprex_del(ComprEx<data_t>* self) { delete self;}
void Comprex_setThreshold(ComprEx<data_t>* self, const ThresholdFunction<data_t>* threshold){
    self->setThreshold(threshold);
}
void Comprex_setCompressor(ComprEx<data_t>* self, const Compressor<data_t>* compressor){
    self->setCompressor(compressor);
}
void Comprex_resetRests(ComprEx<data_t>* self){
    self->resetRests();
}
void Comprex_flushRests(ComprEx<data_t>* self){
    self->flushRests();
}
void Comprex_getRests(ComprEx<data_t>* self, data_t* vector, int size){
        self->getRests(vector, size);
}
void Comprex_writeRemote(ComprEx<data_t>* self, const data_t* vector, int size) {
    self->writeRemote( vector, size);
}
void Comprex_readRemote(ComprEx<data_t>* self, data_t* vector, int size) {
    self->readRemote( vector, size);
}
void Comprex_readRemote_add(ComprEx<data_t>* self, data_t* vector, int size) {
    self->readRemote_add(vector, size);
}
void Comprex_connectTo(ComprEx<data_t>* self, int srcRank, int targRank, int tag){
    self->connectTo(static_cast<gaspi::group::Rank>(srcRank), static_cast<gaspi::group::Rank>(targRank), tag);
}

void Comprex_connectTx(ComprEx<data_t>* self, int targRank, int tag){
    self->connectTx(static_cast<gaspi::group::Rank>(targRank), tag);
}

void Comprex_connectRx(ComprEx<data_t>* self, int srcRank, int tag){
    self->connectRx(static_cast<gaspi::group::Rank>(srcRank), tag);
}

///////////////////////
// Thresholds
//////////////////////
// ThresholdNone<data_t>
ThresholdNone<data_t>* ThresholdNone_new() { return new ThresholdNone<data_t>();}
void ThresholdNone_del(ThresholdNone<data_t>* self) { delete self;}

// ThresholdConst<data_t>
ThresholdConst<data_t>* ThresholdConst_new(data_t n) { return new ThresholdConst<data_t>(n);}
void ThresholdConst_del(ThresholdConst<data_t>* self) { delete self;}
data_t ThresholdConst_getThreshold(ThresholdConst<data_t>* inst) { return inst->get_threshold();}

// ThresholdTopK<data_t>
ThresholdTopK<data_t>* ThresholdTopK_new(float topK) { return new ThresholdTopK<data_t>(topK);}
void ThresholdTopK_del(ThresholdTopK<data_t>* self) { delete self;}
data_t ThresholdTopK_getThreshold(ThresholdTopK<data_t>* inst) { return inst->get_threshold();}

///////////////////////
// Compressors
//////////////////////
// CompressorNone<data_t>
CompressorNone<data_t>* CompressorNone_new() { return new CompressorNone<data_t>();}
void CompressorNone_del(CompressorNone<data_t>* self) { delete self;}

// CompressorRLE<data_t>
CompressorRLE<data_t>* CompressorRLE_new() { return new CompressorRLE<data_t>();}
void CompressorRLE_del(CompressorRLE<data_t>* self) { delete self;}

// CompressorIndexPairs<data_t>
CompressorIndexPairs<data_t>* CompressorIndexPairs_new() { return new CompressorIndexPairs<data_t>();}
void CompressorIndexPairs_del(CompressorIndexPairs<data_t>* self) { delete self;}

} // extern "C"
