#include "compressedExchange.hxx"
#include "compressor.hxx"
#include "threshold.hxx"
#include <stdio.h>
#include <vector>
#include <GASPI_Ext.h>

extern "C"
{

typedef float data_t;

///////////////////////
// ComprEx
//////////////////////
ComprEx<data_t>* Comprex_new( gaspi::Runtime* runTime, gaspi::Context* context, gaspi::segment::Segment* segment, int size) { 
    return new ComprEx<data_t>((*runTime), (*context), (*segment), size);
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
        const std::vector<data_t>* rests = self->getRests_p();
        if(rests->size() != size){
            char error_msg[100];
            sprintf(error_msg,"getRests: Rests of size %d does not match the buffer of size %d!\n", (int)rests->size(), (int)size);
            throw std::runtime_error(error_msg);
        }
        for(int i=0; i<size; ++i){
            vector[i] = (*rests)[i];
        }
}
void Comprex_writeRemote(ComprEx<data_t>* self, const data_t* vector, int size) {
    self->writeRemote( vector, size);
}
void Comprex_readRemote(ComprEx<data_t>* self, data_t* vector, int size) {
    self->readRemote( vector, size);
}

void Comprex_connectTo(ComprEx<data_t>* self, int srcRank, int targRank, int tag, int size_factor){
    self->connectTo(static_cast<gaspi::group::Rank>(srcRank), static_cast<gaspi::group::Rank>(targRank), tag, size_factor);
}

void Comprex_connectTx(ComprEx<data_t>* self, int targRank, int tag, int size_factor){
    self->connectTx(static_cast<gaspi::group::Rank>(targRank), tag, size_factor);
}

void Comprex_connectRx(ComprEx<data_t>* self, int srcRank, int tag, int size_factor){
    self->connectRx(static_cast<gaspi::group::Rank>(srcRank), tag, size_factor);
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
data_t ThresholdConst_getThreshold(ThresholdConst<data_t>* inst) { return inst->getThreshold();}

// ThresholdTopK<data_t>
ThresholdTopK<data_t>* ThresholdTopK_new(float topK) { return new ThresholdTopK<data_t>(topK);}
void ThresholdTopK_del(ThresholdTopK<data_t>* self) { delete self;}
data_t ThresholdTopK_getThreshold(ThresholdTopK<data_t>* inst) { return inst->getThreshold();}

///////////////////////
// Compressors
//////////////////////
// CompressorNone<data_t>
CompressorNone<data_t>* CompressorNone_new() { return new CompressorNone<data_t>();}
void CompressorNone_del(CompressorNone<data_t>* self) { delete self;}

// CompressorRLE<data_t>
CompressorRLE<data_t>* CompressorRLE_new() { return new CompressorRLE<data_t>();}
void CompressorRLE_del(CompressorRLE<data_t>* self) { delete self;}

} // extern "C"
