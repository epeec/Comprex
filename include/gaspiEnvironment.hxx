
#pragma once
/* GaspiEnvironment
* Handles the GaspiCXX runtime setup.
*/

#include <GaspiCxx/segment/Segment.hpp>

class GaspiEnvironment {
public:
    GaspiEnvironment(unsigned long segment_size);

    ~GaspiEnvironment();

    // local rank
    int get_rank();
    // local size
    int get_ranks();

    void barrier();

    void flush();

    gaspi::segment::Segment* get_segment();

private:
    gaspi::segment::Segment* segment;
};
