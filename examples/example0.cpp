/*
 *  example0.cxx
 *
 *
 */

#include <comprexchange.hxx>

main(int argc, char* argv[])
{

  std::string myMessage("Hi");
  comprEx::MyClass myClass(myMessage);

  myClass.printMessage();

  return 0;
}

