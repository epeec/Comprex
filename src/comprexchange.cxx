/*
 *  comprexchange.cxx
 *
 *
 */

#include <comprexchange.hxx>

namespace comprEx {

  MyClass::MyClass(std::string myString)
    :
    _myMsg(myString)
  {

  }

  MyClass::~MyClass() 
  {

  }

  void
  MyClass::printMessage() const
  {
    std::cout << "The message is: " << _myMsg << std::endl; 
  }


} // end namespace comprEx
