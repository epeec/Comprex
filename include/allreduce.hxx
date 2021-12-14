#ifndef ALLREDUCE_HXX
#define ALLREDUCE_HXX

#include <vector>
#include <map>
#include <tuple>
#include <utility>

#include "comprEx.hxx"
#include "gaspiEx.hxx"
#include "gaspiEnvironment.hxx"

/* Baseclass for allreduce operations
*/
template <typename data_t>
class Allreduce_base{
public:
    Allreduce_base(GaspiEnvironment* gaspi_env){
        this->gaspi_env = gaspi_env;
    }

    virtual ~Allreduce_base(){
    }

    virtual void apply(data_t *data, int size) = 0;

protected:
    // GaspiCxx
    GaspiEnvironment* gaspi_env;
};

/* AllToOneAllreduce
* Allreduce which reduces all Data in one node and broadcast data back from one node.
* This implementation does not use Comprex.
*/
template <typename data_t>
class AllToOneAllreduce : public Allreduce_base<data_t>{
public:
    AllToOneAllreduce(GaspiEnvironment* gaspi_env);

    ~AllToOneAllreduce();

    void setupConnections(int size, int tag, int chiefRank);

    virtual void apply(data_t *data, int size);

protected:
    // communicator the other communicators will be copying from in the chief rank
    int copy_comm = -1;
    // size of data that is communicated
    int comm_size;
    // chief rank, which does the reduction operation
    int chiefRank;
    // maps for communication patterns
    std::map<int, GaspiEx<data_t> > comm_reduce_map;
    std::map<int, GaspiEx<data_t> > comm_broadcast_map;
};


/* Comprex_AllToOneAllreduce
* Allreduce which reduces all Data in one node and broadcast data back from one node.
* This implementation uses Comprex to compress data in both the gather and broadcast round.
*/
template <typename data_t>
class Comprex_AllToOneAllreduce : public Allreduce_base<data_t> {
public:
    Comprex_AllToOneAllreduce(GaspiEnvironment* gaspi_env);

    ~Comprex_AllToOneAllreduce();

    void setupConnections(int size, int tag, int chiefRank, float compression_ratio, float size_factor=1.0);

    virtual void apply(data_t *data, int size);

    // flush the comprex communicators internal state, communicating all the updates.
    virtual void flush(data_t *data, int size);

    // the comprex communicators have internal states (i.e. rests vectors), which can be reset by this function to 0.
    void reset();

    // set a new topK Value for the compression
    void set_compression_ratio(float compression_ratio);

protected:
    // maps for communication patterns
    std::map<int, FastComprEx<data_t> > comm_reduce_map;
    std::map<int, FastComprEx<data_t> > comm_broadcast_map;
    // std::map<int, ComprEx<data_t> > comm_reduce_map;
    // std::map<int, ComprEx<data_t> > comm_broadcast_map;
    // size of data that is communicated
    int comm_size;
    // chief rank, which does the reduction operation
    int chiefRank;
};


/* RingAllreduce
* Allreduce which reduces data in a ring and broadcasts data in a ring.
* No compression is used in this version.
*/
template <typename data_t>
class RingAllreduce : public Allreduce_base<data_t> {
public:
    RingAllreduce(GaspiEnvironment* gaspi_env);

    ~RingAllreduce();

    void setupConnections(int size, int tag);

    virtual void apply(data_t *data, int size);

protected:
    // maps for communication patterns
    std::map<int, GaspiEx<data_t> > comm_reduce_map;
    std::map<int, GaspiEx<data_t> > comm_broadcast_map;
    // size of data that is communicated
    int total_size;
    std::vector<int> segment_sizes;
    std::vector<int> segment_starts;
};


/* Comprex_RingAllreduce
* Allreduce which reduces data in a ring and broadcasts data in a ring.
* This implementation uses Comprex to compress data in both the gather and broadcast round.
*/
template <typename data_t>
class Comprex_RingAllreduce : public Allreduce_base<data_t> {
public:
    Comprex_RingAllreduce(GaspiEnvironment* gaspi_env);

    ~Comprex_RingAllreduce();

    void setupConnections(int size, int tag, float compression_ratio, float size_factor=1.0);

    virtual void apply(data_t *data, int size);

    /* flush the comprex communicators internal state, communicating all the updates */
    virtual void flush(data_t *data, int size);

    // the comprex communicators have internal states (i.e. rests vectors), which can be reset by this function to 0.
    void reset();

    // set a new topK Value for the compression
    void set_compression_ratio(float compression_ratio);

protected:
    // maps for communication patterns
    std::map<int, ComprEx<data_t> > comm_reduce_map;
    std::map<int, ComprEx<data_t> > comm_broadcast_map;
    // size of data that is communicated
    int total_size;
    std::vector<int> segment_sizes;
    std::vector<int> segment_starts;
};


/* BigRingAllreduce
* Allreduce which reduces data in a ring, without segmenting it.
*/
template <typename data_t>
class BigRingAllreduce : public Allreduce_base<data_t> {
public:
    BigRingAllreduce(GaspiEnvironment* gaspi_env);

    ~BigRingAllreduce();

    void setupConnections(int size, int tag);

    virtual void apply(data_t *data, int size);

protected:
    // maps for communication patterns
    GaspiEx<data_t> comm_reduce[1];
    GaspiEx<char> comm_feedback[2];
    // size of data that is communicated
    int total_size;
};


/* Comprex_BigRing
* Allreduce which reduces data in a ring, without segmenting it.
* This implementation uses Comprex to compress data in both the gather and broadcast round.
*/
template <typename data_t>
class Comprex_BigRingAllreduce : public Allreduce_base<data_t> {
public:
    Comprex_BigRingAllreduce(GaspiEnvironment* gaspi_env);

    ~Comprex_BigRingAllreduce();

    void setupConnections(int size, int tag, float compression_ratio, float size_factor=1.0);

    virtual void apply(data_t *data, int size);

    /* flush the comprex communicators internal state, communicating all the updates */
    virtual void flush(data_t *data, int size);

    // the comprex communicators have internal states (i.e. rests vectors), which can be reset by this function to 0.
    void reset();

    // set a new topK Value for the compression
    virtual void set_compression_ratio(float compression_ratio);

protected:
    // maps for communication patterns
    ComprEx<data_t> comm_reduce[1];
    GaspiEx<char> comm_feedback[2];
    // size of data that is communicated
    int total_size;
};

#endif //ALLREDUCE_HXX
