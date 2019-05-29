// myRandomGen.hxx : random generator class, definitions

#include <memory>
#include <random>

#ifndef MY_RANDOM_GEN_HXX
#define MY_RANDOM_GEN_HXX

namespace random_generator {

  template <class VarTYPE>
  class MyRandomGen {
    private:
     VarTYPE _lowerBound;
     VarTYPE _upperBound;

     std::random_device _randmDevice;
     std::unique_ptr<std::mt19937> _engine; // standard mersenne twister engine
     std::unique_ptr< std::uniform_real_distribution<VarTYPE> > _realDistr;
     std::unique_ptr< std::uniform_int_distribution<>  > _intDistr;

    public:

      MyRandomGen(VarTYPE lowerBnd, VarTYPE upperBnd);

      ~MyRandomGen();

      VarTYPE generateRandomNumber();

  };// MyRandomGen 

  //#include <myRandomGen.cxx>

} // namespace random generator

#endif // defime MY_RANDOM_GEN_HXX
