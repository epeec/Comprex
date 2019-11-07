#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Allocator.hpp>
#include <GaspiCxx/segment/NotificationManager.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
#include <GaspiCxx/utility/ScopedAllocation.hpp>

#include <GASPI.h>
#include <GASPI_Ext.h>

#include <stdio.h>

typedef float data_t;

extern "C"
{

gaspi::Runtime* Gaspi_Runtime_new() {return new gaspi::Runtime();}
void Gaspi_Runtime_del(gaspi::Runtime* self) {delete self;}

bool Gaspi_isRuntimeAvailable() {return gaspi::isRuntimeAvailable();}

gaspi::Context* Gaspi_Context_new() {return new gaspi::Context();}
void Gaspi_Context_del(gaspi::Context* self) {delete self;}

int Gaspi_Context_getRank(gaspi::Context* context) { return static_cast<int>(context->rank().get()); }

int Gaspi_Context_getSize(gaspi::Context* context) { return static_cast<int>(context->size().get()); }

void Gaspi_Context_barrier(gaspi::Context* context) { context->barrier();}

gaspi::segment::Segment* Gaspi_Segment_new(int size) { return new gaspi::segment::Segment(static_cast<size_t>(size)); }
void Gaspi_Segment_del(gaspi::segment::Segment* self) {delete self;}

void Gaspi_Printf(const char* msg) {
    gaspi_printf("%s\n", msg);
}


void Gaspi_Allreduce_floatsum(float* const output, float* input, int size, gaspi::Context* context) {
    gaspi_allreduce( 
          input, 
          output,
          size,
          GASPI_OP_SUM,
          GASPI_TYPE_FLOAT,
          context->group().group(),
          GASPI_BLOCK );
}

unsigned int Gaspi_Allreduce_Elem_Max() {
    unsigned int elem_max;
    gaspi_allreduce_elem_max (&elem_max);
    return elem_max;
}




} // extern "C"