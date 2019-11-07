// myRandomGen.cxx : random generator class, implementation

#include <myRandomGen.hxx>

#ifndef MY_RANDOM_GEN_CXX
#define MY_RANDOM_GEN_CXX

namespace random_generator {

  template <class VarTYPE>
  MyRandomGen<VarTYPE>::MyRandomGen(VarTYPE lowerBnd, VarTYPE upperBnd)
       :
         _lowerBound(lowerBnd)
       , _upperBound(upperBnd)
	 , _randmDevice()
         , _engine()
         , _realDistr()
     {

       _engine =  std::unique_ptr<std::mt19937>
         (new std::mt19937(_randmDevice()));  
       _realDistr = std::unique_ptr< std::uniform_real_distribution<VarTYPE> >
	 (new std::uniform_real_distribution<VarTYPE> 
                              (_lowerBound, _upperBound) 
         )
;
       _intDistr = std::unique_ptr< std::uniform_int_distribution<> >
        (new std::uniform_int_distribution<> (
	   static_cast<int>(_lowerBound), static_cast<int>(_upperBound)) );

  }

  template <class VarTYPE>
  MyRandomGen<VarTYPE>::~MyRandomGen()
  {

  }

  /*
  // specialization for int
  template <>
  int MyRandomGen<int>::generateRandomNumber()
  {
    printf("\n int variant called.. ");
    return (*_intDistr)(*_engine);
  }
  */

  // all kinds of real numbers: float, double 
  template <class VarTYPE>
  VarTYPE MyRandomGen<VarTYPE>::generateRandomNumber()
  {

        return (*_realDistr)(*_engine);
  }
     
}// namespace random_generator {

#endif
