// ETI.cpp - explicit template instantiation
//
// The code below ensures that templated classes and methods
// are instantiated with certain template parameters.

#include <complex>

// Include the definitions here...
#include <comprex.hxx>
#include <mThrTopKcompress.hxx>

namespace compressed_exchange {

#ifdef ETI_INT
  template class ComprEx<int>;
  template class ComprExRunLengths<int>;  
  template class MultiThreadedRLE<int>;
  template class ComprExTopK<int>;
  template class MultiThreadedTopK<int>;
  template void *thread_func_compressTopK<int>(void *args);
#endif

#ifdef ETI_DOUBLE
  template class ComprEx<double>;
  template class ComprExRunLengths<double>; 
  template class MultiThreadedRLE<double>;
  template class ComprExTopK<double>;
  template class MultiThreadedTopK<double>;
  template void *thread_func_compressTopK<double>(void *args);
#endif

} // namespace compressed_exchange
