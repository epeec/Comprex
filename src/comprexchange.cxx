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
    // a test comment added, to check how .gitignore works
    std::cout << "The message is: " << _myMsg << std::endl; 
  }


} // end namespace comprEx
