/*
 *  comprex.hxx
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

namespace compressed_exchange {

  class MyClass {
     private: 

       std::string & _myMsg;

     public:
    
      MyClass(std::string myString);
      ~MyClass();

      void printMessage() const;

  }; // myClass

  template <class VarTYPE>
  class ComprEx {
       
    private:

       // GaspiCxx 
       gaspi::Runtime & _gpiCxx_runtime;
       gaspi::Context & _gpiCxx_context;
       gaspi::segment::Segment & _gpiCxx_segment;

    public:
       ComprEx( gaspi::Runtime & runTime
              , gaspi::Context & context
	      , gaspi::segment::Segment & segment
	       );

       ~ComprEx();


    // Compress the input vector of type VarTYPE and given size,
    // using the treshold provided as an argument,
    // and then GaspiCxx-send it to the destination rank destRank. 
    void Compress_and_WriteRemote(
	       std::unique_ptr<VarTYPE []> const & vector // pointer to the vector
               , int size                         // vector´s (original) size
	       , VarTYPE treshold                 // treshold
	       , gaspi::group::Rank  destRank);   // destination rank

  }; // ComprEx 
  

}  // end namespace compressed_exchange

// include the template implementation
#include <comprex.cxx>

#endif // #define COMPREX_H
