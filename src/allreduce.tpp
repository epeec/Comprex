#include <vector>
#include <map>
#include <tuple>
#include <utility>

#include <GASPI_Ext.h>

#include "comprEx.hxx"
#include "comprEx.tpp"
#include "gaspiEx.hxx"
#include "gaspiEx.tpp"
#include "threshold.hxx"
#include "compressor.hxx"
#include "gaspiEnvironment.hxx"

/* AllToOneAllreduce
* Allreduce which reduces all Data in one node and broadcast data back from one node.
* This implementation does not use Comprex.
*/
template <typename data_t>
AllToOneAllreduce<data_t>::AllToOneAllreduce(GaspiEnvironment* gaspi_env) 
    : Allreduce_base<data_t>(gaspi_env) {
}

template <typename data_t>
AllToOneAllreduce<data_t>::~AllToOneAllreduce(){
}

template <typename data_t>
void AllToOneAllreduce<data_t>::setupConnections(int size, int tag, int chiefRank){
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    this->chiefRank = chiefRank;
    this->comm_size = size;
    int comprexTag = tag*2*numRanks;
    int commTag = tag*2*numRanks+1;

    auto comm_proto = GaspiEx<data_t>(this->gaspi_env->get_segment(), size);

    // connection for comprex is all to one
    if(myRank==chiefRank) {
        for(int rank=0; rank<numRanks; ++rank) {
            if(rank==chiefRank) continue;
            comm_reduce_map[rank]=comm_proto;
            comm_reduce_map[rank].connectRx(rank, comprexTag+rank);
        }
    } else {
        comm_reduce_map[chiefRank]=comm_proto;
        comm_reduce_map[chiefRank].connectTx(chiefRank, comprexTag+myRank);
    }
        
    // connection for comm is one to all
    if(myRank==chiefRank) {
        for(int rank=0; rank<numRanks; ++rank) {
            if(rank==chiefRank) continue;
            if(copy_comm==-1){
                comm_broadcast_map[rank]=comm_proto;
                comm_broadcast_map[rank].connectTx(rank, commTag+rank);
                copy_comm = rank;
            }
            else {
                comm_broadcast_map[rank]=comm_proto;
                comm_broadcast_map[rank].connectTx_sameBufferAs(rank, commTag+rank, &comm_broadcast_map[copy_comm]);
                // comm_broadcast_map[rank].connectTx(rank, commTag+rank);
            }
        }
    } else {
        comm_broadcast_map[chiefRank]=comm_proto;
        comm_broadcast_map[chiefRank].connectRx(chiefRank, commTag+myRank);
    }
}

template <typename data_t>
void AllToOneAllreduce<data_t>::apply(data_t *data, int size) {
    // check if target vector can hold communicated vector
    if(size < comm_size) {
        char error_msg[100];
        sprintf(error_msg,"Vector size (%d) is smaller than communicated data size (%d)!\n", size, comm_size);
        throw std::runtime_error(error_msg);
    }
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();

    // reduces data
    if(myRank == chiefRank){
        // Chief Rank
        // reduce
        for(int rank=0; rank<numRanks; ++rank) {
            if(rank == chiefRank) continue;
            comm_reduce_map[rank].readRemote_add(data, size);
        }
    } else {
        // Worker Rank
        // send own data
        comm_reduce_map[chiefRank].writeRemote(data, size);
    }

    // broadcast data
    if(myRank == chiefRank){
        // Chief Rank
        // broadcast
        for( int rank=0; rank<numRanks; ++rank) {
            if(rank == chiefRank) continue;
            if(rank == copy_comm) {
                comm_broadcast_map[rank].writeRemote(data, size);
            }
            else {
                comm_broadcast_map[rank].writeRemote(size);
                // comm_broadcast_map[rank].writeRemote(data, size);
            }
            
        }
    } else {
        // Worker Rank
        // receive data
        comm_broadcast_map[chiefRank].readRemote(data, size);
    }
}

/* Comprex_AllToOneAllreduce
* Allreduce which reduces all Data in one node and broadcast data back from one node.
* This implementation uses Comprex to compress data in both the gather and broadcast round.
*/
template <typename data_t>
Comprex_AllToOneAllreduce<data_t>::Comprex_AllToOneAllreduce(GaspiEnvironment* gaspi_env) 
        : Allreduce_base<data_t>(gaspi_env) {
}

template <typename data_t>
Comprex_AllToOneAllreduce<data_t>::~Comprex_AllToOneAllreduce(){
}

template <typename data_t>
void Comprex_AllToOneAllreduce<data_t>::setupConnections(int size, int tag, int chiefRank, float compression_ratio, float size_factor){
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    this->chiefRank = chiefRank;
    this->comm_size = size;
    int comprexTag = tag*2*numRanks;
    int commTag = tag*2*numRanks+1;

    auto comm_proto = FastComprEx<data_t>(this->gaspi_env->get_segment(), size, size_factor);
    comm_proto.setThreshold(ThresholdTopK<data_t>(compression_ratio));
    comm_proto.setCompressor(CompressorIndexPairs<data_t>());

    // connection for comprex is all to one
    if(myRank==chiefRank) {
        for(int rank=0; rank<numRanks; ++rank) {
            if(rank==chiefRank) continue;
            comm_reduce_map[rank]=comm_proto;
            comm_reduce_map[rank].connectRx(rank, comprexTag+rank);
        }
    } else {
        comm_reduce_map[chiefRank]=comm_proto;
        comm_reduce_map[chiefRank].connectTx(chiefRank, comprexTag+myRank);
    }
        
    // connection for comm is one to all
    if(myRank==chiefRank) {
        for(int rank=0; rank<numRanks; ++rank) {
            if(rank==chiefRank) continue;
            comm_broadcast_map[rank]=comm_proto;
            comm_broadcast_map[rank].connectTx(rank, commTag+rank);
        }
    } else {
        comm_broadcast_map[chiefRank]=comm_proto;
        comm_broadcast_map[chiefRank].connectRx(chiefRank, commTag+myRank);
    }
}

template <typename data_t>
void Comprex_AllToOneAllreduce<data_t>::apply(data_t *data, int size) {
    // check if target vector can hold communicated vector
    if(size < comm_size) {
        char error_msg[100];
        sprintf(error_msg,"Vector size (%d) is smaller than communicated data size (%d)!\n", size, comm_size);
        throw std::runtime_error(error_msg);
    }
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();

    // reduces data
    if(myRank == chiefRank){
        // Chief Rank
        // reduce
        for(int rank=0; rank<numRanks; ++rank) {
            if(rank == chiefRank) continue;
            comm_reduce_map[rank].readRemote_add(data, size);
        }
    } else {
        // Worker Rank
        // send own data
        comm_reduce_map[chiefRank].writeRemote(data, size);
    }

    // broadcast data
    if(myRank == chiefRank){
        // Chief Rank
        int copy_rank = -1;
        
        // broadcast
        for( int rank=0; rank<numRanks; ++rank) {
            if(rank == chiefRank) continue;
            if(copy_rank == -1){
                comm_broadcast_map[rank].writeRemote(data, size);
                copy_rank = rank;
            }
            else {
                comm_broadcast_map[rank].writeRemote_sameAs(&comm_broadcast_map[copy_rank]);
            }
        }
    } else {
        // Worker Rank
        // receive data
        comm_broadcast_map[chiefRank].readRemote(data, size);
    }
}

template <typename data_t>
void Comprex_AllToOneAllreduce<data_t>::flush(data_t *data, int size) {
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    // reduces data
    if(myRank == chiefRank){
        // Chief Rank
        // reduce
        for(int rank=0; rank<numRanks; ++rank) {
            if(rank == chiefRank) continue;
            comm_reduce_map[rank].readRemote_add(data, size);
        }
    } else {
        // Worker Rank
        // send Rests
        comm_reduce_map[chiefRank].flushRests();
    }
    // do not broadcast data back! Reset all communicators.
    reset();
}

template <typename data_t>
void Comprex_AllToOneAllreduce<data_t>::reset(){
    for (auto &comm : comm_reduce_map) {
        comm.second.resetRests();
    }
    for (auto &comm : comm_broadcast_map) {
        comm.second.resetRests();
    }
}

template <typename data_t>
void Comprex_AllToOneAllreduce<data_t>::set_compression_ratio(float compression_ratio){
    for (auto &comm : comm_reduce_map) {
        comm.second.setThreshold(ThresholdTopK<data_t>(compression_ratio));
    }
    for (auto &comm : comm_broadcast_map) {
        comm.second.setThreshold(ThresholdTopK<data_t>(compression_ratio));
    }
}

/* RingAllreduce
* Allreduce which reduces data in a ring and broadcasts data in a ring.
* No compression is used in this version.
*/
template <typename data_t>
RingAllreduce<data_t>::RingAllreduce(GaspiEnvironment* gaspi_env)
    : Allreduce_base<data_t>(gaspi_env) {
}

template <typename data_t>
RingAllreduce<data_t>::~RingAllreduce(){
}

template <typename data_t>
void RingAllreduce<data_t>::setupConnections(int size, int tag){
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    this->total_size = size;

    int segment_size = size / numRanks;
    segment_sizes.resize(numRanks);
    for(int it=0; it<numRanks; ++it){
        segment_sizes[it] = segment_size;
    }

    int segment_residual = size % numRanks;
    for(int it=0; it<segment_residual; ++it){
        segment_sizes[it] += 1;
    }

    // Compute where each chunk starts.
    segment_starts.resize(numRanks);
    segment_starts[0] = 0;
    for(int it=1; it<numRanks; ++it){ 
        segment_starts[it] = segment_sizes[it-1]+segment_starts[it-1];
    }

    // assert: The last segment should end at the very end of the buffer.
    if(segment_starts[numRanks-1]+segment_sizes[numRanks-1] != total_size){
        throw std::runtime_error("Last segment should end at the total size of the data!");
    }

    // Receive from your left neighbor with wrap-around.
    int recv_from = (myRank - 1 + numRanks) % numRanks;
    // Send to your right neighbor with wrap-around.
    int send_to = (myRank + 1) % numRanks;

    for(int chunk=0; chunk<numRanks; ++chunk){
        int reduce_tag = tag+chunk;
        int broadcast_tag = tag+numRanks+chunk;
        // reduce ring
        comm_reduce_map[chunk] = GaspiEx<data_t>(this->gaspi_env->get_segment(), segment_sizes[chunk]);
        comm_reduce_map[chunk].connectTo(recv_from, send_to, reduce_tag);
        // broadcast ring
        comm_broadcast_map[chunk] = GaspiEx<data_t>(this->gaspi_env->get_segment(), segment_sizes[chunk]);
        comm_broadcast_map[chunk].connectTo(recv_from, send_to, broadcast_tag);
    }
}

template <typename data_t>
void RingAllreduce<data_t>::apply(data_t *data, int size) {
    // check if target vector can hold communicated vector
    if(size < this->total_size) {
        char error_msg[100];
        sprintf(error_msg,"Vector size (%d) is smaller than communicated data size (%d)!\n", size, this->total_size);
        throw std::runtime_error(error_msg);
    }
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    int recv_chunk, send_chunk;
    data_t *recv_segment, *send_segment;

    // reduction
    for(int it=0; it<numRanks-1; ++it) {
        recv_chunk = (myRank - it - 1 + numRanks) % numRanks;
        send_chunk = (myRank - it + numRanks) % numRanks;
        recv_segment = data+segment_starts[recv_chunk];
        send_segment = data+segment_starts[send_chunk];
        comm_reduce_map[send_chunk].writeRemote(send_segment, segment_sizes[send_chunk]);
        comm_reduce_map[recv_chunk].readRemote_add(recv_segment, segment_sizes[recv_chunk]);
    }

    // broadcast
    for(int it=0; it<numRanks-1; ++it) {
        recv_chunk = (myRank - it + numRanks) % numRanks;
        send_chunk = (myRank - it + 1 + numRanks) % numRanks;
        recv_segment = data+segment_starts[recv_chunk];
        send_segment = data+segment_starts[send_chunk];
        comm_broadcast_map[send_chunk].writeRemote(send_segment, segment_sizes[send_chunk]);
        comm_broadcast_map[recv_chunk].readRemote(recv_segment, segment_sizes[recv_chunk]);
    }
}

/* Comprex_RingAllreduce
* Allreduce which reduces data in a ring and broadcasts data in a ring.
* This implementation uses Comprex to compress data in both the gather and broadcast round.
*/
template <typename data_t>
Comprex_RingAllreduce<data_t>::Comprex_RingAllreduce(GaspiEnvironment* gaspi_env)
    : Allreduce_base<data_t>(gaspi_env) {
}

template <typename data_t>
Comprex_RingAllreduce<data_t>::~Comprex_RingAllreduce(){
}

template <typename data_t>
void Comprex_RingAllreduce<data_t>::setupConnections(int size, int tag, float compression_ratio, float size_factor){
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    this->total_size = size;

    int segment_size = size / numRanks;
    segment_sizes.resize(numRanks);
    for(int it=0; it<numRanks; ++it){
        segment_sizes[it] = segment_size;
    }

    int segment_residual = size % numRanks;
    for(int it=0; it<segment_residual; ++it){
        segment_sizes[it] += 1;
    }

    // Compute where each chunk starts.
    segment_starts.resize(numRanks);
    segment_starts[0] = 0;
    for(int it=1; it<numRanks; ++it){ 
        segment_starts[it] = segment_sizes[it-1]+segment_starts[it-1];
    }

    // assert: The last segment should end at the very end of the buffer.
    if(segment_starts[numRanks-1]+segment_sizes[numRanks-1] != total_size){
        throw std::runtime_error("Last segment should end at the total size of the data!");
    }

    // Receive from your left neighbor with wrap-around.
    int recv_from = (myRank - 1 + numRanks) % numRanks;
    // Send to your right neighbor with wrap-around.
    int send_to = (myRank + 1) % numRanks;

    // flag for attaching size of bytes to comprex messages. This allows for faster communication.
    bool send_sizeBytes=true;
    int thresholdTopK_subsamples=1000/numRanks;
    float thresholdTopK_sampleFactor=4.0;
    for(int chunk=0; chunk<numRanks; ++chunk){
        int reduce_tag = tag+chunk;
        int broadcast_tag = tag+numRanks+chunk;
        // reduce ring
        comm_reduce_map[chunk] = ComprEx<data_t>(this->gaspi_env->get_segment(), segment_sizes[chunk], size_factor, send_sizeBytes);
        comm_reduce_map[chunk].setThreshold(ThresholdTopK<data_t>(compression_ratio, thresholdTopK_subsamples, thresholdTopK_sampleFactor));
        comm_reduce_map[chunk].setCompressor(CompressorIndexPairs<data_t>());
        comm_reduce_map[chunk].connectTo(recv_from, send_to, reduce_tag);
        // broadcast ring
        comm_broadcast_map[chunk] = ComprEx<data_t>(this->gaspi_env->get_segment(), segment_sizes[chunk], size_factor, send_sizeBytes);
        comm_broadcast_map[chunk].setThreshold(ThresholdTopK<data_t>(compression_ratio));
        comm_broadcast_map[chunk].setCompressor(CompressorIndexPairs<data_t>());
        comm_broadcast_map[chunk].connectTo(recv_from, send_to, broadcast_tag);
    }
}

template <typename data_t>
void Comprex_RingAllreduce<data_t>::apply(data_t *data, int size) {
    // check if target vector can hold communicated vector
    if(size < this->total_size) {
        char error_msg[100];
        sprintf(error_msg,"Vector size (%d) is smaller than communicated data size (%d)!\n", size, this->total_size);
        throw std::runtime_error(error_msg);
    }
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    int recv_chunk, send_chunk;
    data_t *recv_segment, *send_segment;

    // reduction
    for(int it=0; it<numRanks-1; ++it) {
        recv_chunk = (myRank - it - 1 + numRanks) % numRanks;
        send_chunk = (myRank - it + numRanks) % numRanks;
        recv_segment = data+segment_starts[recv_chunk];
        send_segment = data+segment_starts[send_chunk];
        comm_reduce_map[send_chunk].writeRemote(send_segment, segment_sizes[send_chunk]);
        comm_reduce_map[recv_chunk].readRemote_add(recv_segment, segment_sizes[recv_chunk]);
    }

    // broadcast
    // first round with thresholding
    send_chunk = (myRank + 1 + numRanks) % numRanks;
    send_segment = data+segment_starts[send_chunk];
    comm_broadcast_map[send_chunk].writeRemote_NoResidual(send_segment, segment_sizes[send_chunk]);
    for(int it=0; it<numRanks-2; ++it) {
        recv_chunk = (myRank - it + numRanks) % numRanks;
        recv_segment = data+segment_starts[recv_chunk];
        comm_broadcast_map[recv_chunk].readRemote_forward(recv_segment, segment_sizes[recv_chunk]);
    }
    // final receive
    recv_chunk = (myRank + 2) % numRanks;
    recv_segment = data+segment_starts[recv_chunk];
    comm_broadcast_map[recv_chunk].readRemote(recv_segment, segment_sizes[recv_chunk]);
}

template <typename data_t>
void Comprex_RingAllreduce<data_t>::flush(data_t *data, int size) {
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    int recv_chunk, send_chunk;
    data_t *recv_segment, *send_segment;

    // reduction
    // reduction must add up all rests from all ranks to the segment (myRank+1)
    recv_chunk = (myRank - 0 - 1 + numRanks) % numRanks;
    send_chunk = (myRank - 0 + numRanks) % numRanks;
    recv_segment = data+segment_starts[recv_chunk];
    send_segment = data+segment_starts[send_chunk];
    comm_reduce_map[send_chunk].getRests(send_segment, segment_sizes[send_chunk]);
    comm_reduce_map[send_chunk].resetRests();
    comm_reduce_map[send_chunk].writeRemote_compressOnly(send_segment, segment_sizes[send_chunk]);
    if(numRanks>2) {
        comm_reduce_map[recv_chunk].readRemote(recv_segment, segment_sizes[recv_chunk]);
    }
    else {
        // in the case of only two ranks, the other rank has the final reduced segment.
        comm_reduce_map[recv_chunk].readRemote_add(recv_segment, segment_sizes[recv_chunk]);
    }
    for(int it=1; it<numRanks-1; ++it) {
        recv_chunk = (myRank - it - 1 + numRanks) % numRanks;
        send_chunk = (myRank - it + numRanks) % numRanks;
        recv_segment = data+segment_starts[recv_chunk];
        send_segment = data+segment_starts[send_chunk];
        comm_reduce_map[send_chunk].addRests(send_segment, segment_sizes[send_chunk]);
        comm_reduce_map[send_chunk].resetRests();
        comm_reduce_map[send_chunk].writeRemote_compressOnly(send_segment, segment_sizes[send_chunk]);
        if(it==numRanks-2){
            comm_reduce_map[recv_chunk].readRemote_add(recv_segment, segment_sizes[recv_chunk]);
        }
        else{
            comm_reduce_map[recv_chunk].readRemote(recv_segment, segment_sizes[recv_chunk]);
        }
    }

    // broadcast
    for(int it=0; it<numRanks-1; ++it) {
        recv_chunk = (myRank - it + numRanks) % numRanks;
        send_chunk = (myRank - it + 1 + numRanks) % numRanks;
        recv_segment = data+segment_starts[recv_chunk];
        send_segment = data+segment_starts[send_chunk];
        comm_broadcast_map[send_chunk].writeRemote_compressOnly(send_segment, segment_sizes[send_chunk]);
        comm_broadcast_map[recv_chunk].readRemote(recv_segment, segment_sizes[recv_chunk]);
    }
    reset();
}

template <typename data_t>
void Comprex_RingAllreduce<data_t>::reset(){
    for (auto &comm : comm_reduce_map) {
        comm.second.resetRests();
    }
    for (auto &comm : comm_broadcast_map) {
        comm.second.resetRests();
    }
}

template <typename data_t>
void Comprex_RingAllreduce<data_t>::set_compression_ratio(float compression_ratio){
    for (auto &comm : comm_reduce_map) {
        comm.second.setThreshold(ThresholdTopK<data_t>(compression_ratio));
    }
    for (auto &comm : comm_broadcast_map) {
        comm.second.setThreshold(ThresholdTopK<data_t>(compression_ratio));
    }
}

/* BigRingAllreduce
* Allreduce which reduces data in a ring, without segmenting it.
*/
template <typename data_t>
BigRingAllreduce<data_t>::BigRingAllreduce(GaspiEnvironment* gaspi_env)
    : Allreduce_base<data_t>(gaspi_env) {
}

template <typename data_t>
BigRingAllreduce<data_t>::~BigRingAllreduce(){
}

template <typename data_t>
void BigRingAllreduce<data_t>::setupConnections(int size, int tag){
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    this->total_size = size;

    // Receive from your left neighbor with wrap-around.
    int recv_from = (myRank - 1 + numRanks) % numRanks;
    // Send to your right neighbor with wrap-around.
    int send_to = (myRank + 1) % numRanks;

    // flag for attaching size of bytes to comprex messages. This allows for faster communication.

    int reduce_tag = tag;
    int feedback_tag = tag+numRanks;
    // reduce ring
    for(int it=0; it<1; ++it){
        comm_reduce[it] = GaspiEx<data_t>(this->gaspi_env->get_segment(), size);
        comm_reduce[it].connectTo(recv_from, send_to, reduce_tag+it);
    }
    // broadcast ring
    for(int it=0; it<2; ++it){
        comm_feedback[it] = GaspiEx<char>(this->gaspi_env->get_segment(), 1);
        // reverse send and receive!
        comm_feedback[it].connectTo(send_to, recv_from, feedback_tag+it);
    }
    
}

template <typename data_t>
void BigRingAllreduce<data_t>::apply(data_t *data, int size) {
    // check if target vector can hold communicated vector
    if(size < this->total_size) {
        char error_msg[100];
        sprintf(error_msg,"Vector size (%d) is smaller than communicated data size (%d)!\n", size, this->total_size);
        throw std::runtime_error(error_msg);
    }
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();

    char feedback_token = '0';

    // initial compression and send
    comm_reduce[0].writeRemote(data, size);
    // reduction
    for(int it=0; it<numRanks-2; ++it) {
        comm_reduce[0].readRemote_wait();
        // forwarding the buffer needs two synchronizations:
        // 1st: to signalize the source rank, that data has arrived, so it can start forwarding its buffer.
        // 2nd: copying is finished, so signalize the source rank that it can start sending.
        comm_feedback[0].writeRemote(&feedback_token, 1);
        comm_feedback[0].readRemote(&feedback_token, 1);
        comm_reduce[0].forward_buffer();
        comm_feedback[1].writeRemote(&feedback_token, 1);
        comm_feedback[1].readRemote(&feedback_token, 1);
        comm_reduce[0].forward_add(data, size);
    }
    // last receive round
    comm_reduce[0].readRemote_add(data, size);
}

/* Comprex_BigRing
* Allreduce which reduces data in a ring, without segmenting it.
* This implementation uses Comprex to compress data in both the gather and broadcast round.
*/
template <typename data_t>
Comprex_BigRingAllreduce<data_t>::Comprex_BigRingAllreduce(GaspiEnvironment* gaspi_env)
    : Allreduce_base<data_t>(gaspi_env) {
}

template <typename data_t>
Comprex_BigRingAllreduce<data_t>::~Comprex_BigRingAllreduce(){
}

template <typename data_t>
void Comprex_BigRingAllreduce<data_t>::setupConnections(int size, int tag, float compression_ratio, float size_factor){
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();
    this->total_size = size;

    // Receive from your left neighbor with wrap-around.
    int recv_from = (myRank - 1 + numRanks) % numRanks;
    // Send to your right neighbor with wrap-around.
    int send_to = (myRank + 1) % numRanks;

    // flag for attaching size of bytes to comprex messages. This allows for faster communication.
    bool send_sizeBytes=true;
    int thresholdTopK_subsamples=1000/numRanks;
    float thresholdTopK_sampleFactor=4.0;

    int reduce_tag = tag;
    int feedback_tag = tag+numRanks;
    // reduce ring
    for(int it=0; it<1; ++it){
        comm_reduce[it] = ComprEx<data_t>(this->gaspi_env->get_segment(), size, size_factor, send_sizeBytes);
        comm_reduce[it].setThreshold(ThresholdTopK<data_t>(compression_ratio, thresholdTopK_subsamples, thresholdTopK_sampleFactor));
        comm_reduce[it].setCompressor(CompressorIndexPairs<data_t>());
        comm_reduce[it].connectTo(recv_from, send_to, reduce_tag+it);
    }
    // broadcast ring
    for(int it=0; it<2; ++it){
        comm_feedback[it] = GaspiEx<char>(this->gaspi_env->get_segment(), 1);
        // reverse send and receive!
        comm_feedback[it].connectTo(send_to, recv_from, feedback_tag+it);
    }
}

template <typename data_t>
void Comprex_BigRingAllreduce<data_t>::apply(data_t *data, int size) {
    // check if target vector can hold communicated vector
    if(size < this->total_size) {
        char error_msg[100];
        sprintf(error_msg,"Vector size (%d) is smaller than communicated data size (%d)!\n", size, this->total_size);
        throw std::runtime_error(error_msg);
    }
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();

    char feedback_token = '0';

    // initial compression and send
    comm_reduce[0].writeRemote(data, size);
    // reduction
    for(int it=0; it<numRanks-2; ++it) {
        comm_reduce[0].readRemote_wait();
        // forwarding the buffer needs two synchronizations:
        // 1st: to signalize the source rank, that data has arrived, so it can start forwarding its buffer.
        // 2nd: copying is finished, so signalize the source rank that it can start sending.
        comm_feedback[0].writeRemote(&feedback_token, 1);
        comm_feedback[0].readRemote(&feedback_token, 1);
        comm_reduce[0].forward_buffer();
        comm_feedback[1].writeRemote(&feedback_token, 1);
        comm_feedback[1].readRemote(&feedback_token, 1);
        comm_reduce[0].forward_add(data, size);
    }
    // last receive round
    comm_reduce[0].readRemote_add(data, size);
    //comm_feedback.writeRemote(&feedback_token, 1);
    //comm_feedback.readRemote(&feedback_token, 1);
}

template <typename data_t>
void Comprex_BigRingAllreduce<data_t>::flush(data_t *data, int size) {
    // check if target vector can hold communicated vector
    if(size < this->total_size) {
        char error_msg[100];
        sprintf(error_msg,"Vector size (%d) is smaller than communicated data size (%d)!\n", size, this->total_size);
        throw std::runtime_error(error_msg);
    }
    int myRank = this->gaspi_env->get_rank();
    int numRanks = this->gaspi_env->get_ranks();

    char feedback_token = '0';

    // initial compression and send
    comm_reduce[0].flushRests();
    // reduction
    for(int it=0; it<numRanks-2; ++it) {
        comm_reduce[0].readRemote_wait();
        comm_feedback[0].writeRemote(&feedback_token, 1);
        comm_feedback[0].readRemote(&feedback_token, 1);
        comm_reduce[0].forward_buffer();
        comm_feedback[1].writeRemote(&feedback_token, 1);
        comm_feedback[1].readRemote(&feedback_token, 1);
        comm_reduce[0].forward_add(data, size);
    }
    // last receive round
    comm_reduce[0].readRemote_add(data, size);
    reset();
}

template <typename data_t>
void Comprex_BigRingAllreduce<data_t>::reset(){
    comm_reduce[0].resetRests();
}

template <typename data_t>
void Comprex_BigRingAllreduce<data_t>::set_compression_ratio(float compression_ratio){
    comm_reduce[0].setThreshold(ThresholdTopK<data_t>(compression_ratio));
}
