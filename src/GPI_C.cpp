#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>

#include "gaspiEnvironment.tpp"

typedef float data_t;

extern "C"
{

/////////////////////////////////
// Gaspi Cxx Functions
/////////////////////////////////

// GaspiCxx Runtime
gaspi::Runtime* Gaspi_Runtime_new() {return &gaspi::getRuntime();}

void Gaspi_Runtime_del(gaspi::Runtime* self) {/* do nothing */}

int Gaspi_Runtime_getGlobalRank(gaspi::Runtime* self) { return static_cast<int>(self->global_rank()); }

int Gaspi_Runtime_getGlobalSize(gaspi::Runtime* self) { return static_cast<int>(self->size()); }

void Gaspi_Runtime_barrier(gaspi::Runtime* self) { self->barrier(); }

void Gaspi_Runtime_flush(gaspi::Runtime* self) { self->flush(); }


// GaspiCxx Segment
gaspi::segment::Segment* Gaspi_Segment_new(int size) { return new gaspi::segment::Segment(static_cast<size_t>(size)); }

void Gaspi_Segment_del(gaspi::segment::Segment* self) {delete self;}


// initGaspiCxx
void Gaspi_initGaspiCxx() { gaspi::initGaspiCxx(); }


/////////////////////////////////
// GaspiEnvironment
/////////////////////////////////
GaspiEnvironment* GaspiEnvironment_new(int segment_size){
    return new GaspiEnvironment(segment_size);
}

void GaspiEnvironment_del(GaspiEnvironment* self){
    delete self;
}

int GaspiEnvironment_get_rank(GaspiEnvironment* self) { 
    return self->get_rank();
}

int GaspiEnvironment_get_ranks(GaspiEnvironment* self) { 
    return self->get_ranks();
}

void GaspiEnvironment_barrier(GaspiEnvironment* self) {
    self->barrier();
}

void GaspiEnvironment_flush(GaspiEnvironment* self) { 
    self->flush();
}

/////////////////////////////////
// GPI Functions
/////////////////////////////////
void Gaspi_Printf(const char* msg) {
    gaspi_printf("%s\n", msg);
}


} // extern "C"