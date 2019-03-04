/*
 *  comprexchange.hxx
 *
 *
 */

#include <iostream>
#include <string>

#ifndef COMPREX_H
#define COMPREX_H

namespace comprEx {

  class MyClass {
     private: 

       std::string & _myMsg;

     public:
    
      MyClass(std::string myString);
      ~MyClass();

      void printMessage() const;

  }; // myClass


}  // end namespace comprEx

#endif // #define COMPREX_H
