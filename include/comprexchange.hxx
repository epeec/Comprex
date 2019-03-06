/*
 *  comprexchange.hxx
 *
 *
 */

#include <iostream>
#include <string>

#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>
#include <GaspiCxx/utility/ScopedAllocation.hpp>

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
