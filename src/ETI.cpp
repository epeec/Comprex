// ETI.cpp - explicit template instantiation
//
// The code below ensures that templated classes and methods
// are instantiated with certain template parameters.

#include <complex>

// Include the definitions here...
#include <comprex.hxx>

namespace compressed_exchange {

#ifdef ETI_INT
  template class ComprEx<int>;
  template class ComprExRunLengths<int>;  
  template class MultiThreadedRLE<int>;
  template class ComprExTopK<int>;
#endif

#ifdef ETI_DOUBLE
  template class ComprEx<double>;
  template class ComprExRunLengths<double>; 
  template class MultiThreadedRLE<double>;
  template class ComprExTopK<double>;
#endif

} // namespace compressed_exchange
