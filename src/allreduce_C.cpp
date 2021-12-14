#include "allreduce.hxx"
#include "allreduce.tpp"

extern "C"
{

typedef float data_t;

///////////////////////
// AllToOneAllreduce
//////////////////////
AllToOneAllreduce<data_t>* AllToOneAllreduce_new(GaspiEnvironment* gaspi_env) { 
    return new AllToOneAllreduce<data_t>(gaspi_env);
}
void AllToOneAllreduce_del(AllToOneAllreduce<data_t>* self) { delete self;}
void AllToOneAllreduce_setupConnections(AllToOneAllreduce<data_t>* self, int size, int tag, int chiefRank) {
    self->setupConnections(size, tag, chiefRank);
}
void AllToOneAllreduce_apply(AllToOneAllreduce<data_t>* self, data_t *data, int size){
    self->apply(data, size);
}

///////////////////////
// Comprex_AllToOneAllreduce
//////////////////////
Comprex_AllToOneAllreduce<data_t>* Comprex_AllToOneAllreduce_new(GaspiEnvironment* gaspi_env) { 
    return new Comprex_AllToOneAllreduce<data_t>(gaspi_env);
}
void Comprex_AllToOneAllreduce_del(Comprex_AllToOneAllreduce<data_t>* self) { delete self;}
void Comprex_AllToOneAllreduce_setupConnections(Comprex_AllToOneAllreduce<data_t>* self, int size, int tag, int chiefRank, float compression_ratio, float size_factor) {
    self->setupConnections(size, tag, chiefRank, compression_ratio, size_factor);
}
void Comprex_AllToOneAllreduce_apply(Comprex_AllToOneAllreduce<data_t>* self, data_t *data, int size){
    self->apply(data, size);
}
void Comprex_AllToOneAllreduce_flush(Comprex_AllToOneAllreduce<data_t>* self, data_t *data, int size){
    self->flush(data, size);
}
void Comprex_AllToOneAllreduce_reset(Comprex_AllToOneAllreduce<data_t>* self){
    self->reset();
}

void Comprex_AllToOneAllreduce_set_compression_ratio(Comprex_AllToOneAllreduce<data_t>* self, float compression_ratio){
    self->set_compression_ratio(compression_ratio);
}

///////////////////////
// RingAllreduce
//////////////////////
RingAllreduce<data_t>* RingAllreduce_new(GaspiEnvironment* gaspi_env) { 
    return new RingAllreduce<data_t>(gaspi_env);
}
void RingAllreduce_del(RingAllreduce<data_t>* self) { delete self;}
void RingAllreduce_setupConnections(RingAllreduce<data_t>* self, int size, int tag) {
    self->setupConnections(size, tag);
}
void RingAllreduce_apply(RingAllreduce<data_t>* self, data_t *data, int size){
    self->apply(data, size);
}

///////////////////////
// Comprex_RingAllreduce
//////////////////////
Comprex_RingAllreduce<data_t>* Comprex_RingAllreduce_new(GaspiEnvironment* gaspi_env) { 
    return new Comprex_RingAllreduce<data_t>(gaspi_env);
}
void Comprex_RingAllreduce_del(Comprex_RingAllreduce<data_t>* self) { delete self;}
void Comprex_RingAllreduce_setupConnections(Comprex_RingAllreduce<data_t>* self, int size, int tag, float compression_ratio, float size_factor) {
    self->setupConnections(size, tag, compression_ratio, size_factor);
}
void Comprex_RingAllreduce_apply(Comprex_RingAllreduce<data_t>* self, data_t *data, int size){
    self->apply(data, size);
}
void Comprex_RingAllreduce_flush(Comprex_RingAllreduce<data_t>* self, data_t *data, int size){
    self->flush(data, size);
}
void Comprex_RingAllreduce_reset(Comprex_RingAllreduce<data_t>* self){
    self->reset();
}

void Comprex_RingAllreduce_set_compression_ratio(Comprex_RingAllreduce<data_t>* self, float compression_ratio){
    self->set_compression_ratio(compression_ratio);
}

///////////////////////
// BigRingAllreduce
//////////////////////
BigRingAllreduce<data_t>* BigRingAllreduce_new(GaspiEnvironment* gaspi_env) { 
    return new BigRingAllreduce<data_t>(gaspi_env);
}
void BigRingAllreduce_del(BigRingAllreduce<data_t>* self) { delete self;}
void BigRingAllreduce_setupConnections(BigRingAllreduce<data_t>* self, int size, int tag) {
    self->setupConnections(size, tag);
}
void BigRingAllreduce_apply(BigRingAllreduce<data_t>* self, data_t *data, int size){
    self->apply(data, size);
}


///////////////////////
// Comprex_BigRingAllreduce
//////////////////////
Comprex_BigRingAllreduce<data_t>* Comprex_BigRingAllreduce_new(GaspiEnvironment* gaspi_env) { 
    return new Comprex_BigRingAllreduce<data_t>(gaspi_env);
}
void Comprex_BigRingAllreduce_del(Comprex_BigRingAllreduce<data_t>* self) { delete self;}
void Comprex_BigRingAllreduce_setupConnections(Comprex_BigRingAllreduce<data_t>* self, int size, int tag, float compression_ratio, float size_factor) {
    self->setupConnections(size, tag, compression_ratio, size_factor);
}
void Comprex_BigRingAllreduce_apply(Comprex_BigRingAllreduce<data_t>* self, data_t *data, int size){
    self->apply(data, size);
}
void Comprex_BigRingAllreduce_flush(Comprex_BigRingAllreduce<data_t>* self, data_t *data, int size){
    self->flush(data, size);
}
void Comprex_BigRingAllreduce_reset(Comprex_BigRingAllreduce<data_t>* self){
    self->reset();
}

void Comprex_BigRingAllreduce_set_compression_ratio(Comprex_BigRingAllreduce<data_t>* self, float compression_ratio){
    self->set_compression_ratio(compression_ratio);
}

} // extern C