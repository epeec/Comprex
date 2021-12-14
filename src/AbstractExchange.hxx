#ifndef ABSTRACTEXCHANGE_HXX
#define ABSTRACTEXCHANGE_HXX

#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/group/Rank.hpp>

template<typename VarTYPE>
class AbstractExchange {
public:
    virtual void writeRemote( const VarTYPE* vector, int size) = 0;

    virtual void readRemote( VarTYPE* vector, int size) = 0;

    virtual void readRemote_add( VarTYPE* vector, int size) = 0;

    virtual void connectTx(gaspi::group::Rank targRank, int tag) = 0;

    virtual void connectRx(gaspi::group::Rank srcRank, int tag) = 0;

    // srcRank: receive data from this rank
    // targRank: send data to this rank
    virtual void connectTo(gaspi::group::Rank srcRank, gaspi::group::Rank targRank, int tag) = 0;
};
#endif //ABSTRACTEXCHANGE_HXX