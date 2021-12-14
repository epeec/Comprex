// GaspiCxx
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Segment.hpp>

#include "gaspiEnvironment.hxx"

#include <GASPI_Ext.h>

/* GaspiEnvironment
* Handles the GaspiCXX runtime setup.
*/

GaspiEnvironment::GaspiEnvironment(unsigned long segment_size) {
        // initialize GaspiCxx
        gaspi::initGaspiCxx();
        segment = new gaspi::segment::Segment(segment_size);
    }

GaspiEnvironment::~GaspiEnvironment(){
        delete segment;
    }

int GaspiEnvironment::get_rank() { 
        return static_cast<int>(gaspi::group::Group().rank().get());
    }

int GaspiEnvironment::get_ranks() { 
        return static_cast<int>(gaspi::group::Group().size()); 
    }

void GaspiEnvironment::barrier() {
        gaspi::getRuntime().barrier();
    }

void GaspiEnvironment::flush() { 
        gaspi::getRuntime().flush();
    }

gaspi::segment::Segment* GaspiEnvironment::get_segment(){
        return segment;
    }