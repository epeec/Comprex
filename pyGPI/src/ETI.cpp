// ETI.cpp - explicit template instantiation
//
// The code below ensures that templated classes and methods
// are instantiated with certain template parameters.

#include <complex>

// Include the definitions here...
#include <comprex.hxx>

#define ETI_COMPREX(TYPE) \
template class ComprEx<TYPE>; \
template class CompressorNone<TYPE>; \
template class CompressorRLE<TYPE>; \
template class ThresholdNone<TYPE>; \
template class ThresholdConst<TYPE>; \
template class ThresholdTopK<TYPE>;


ETI_COMPREX(int)
ETI_COMPREX(float)
ETI_COMPREX(double)

