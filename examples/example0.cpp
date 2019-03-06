/*
 *  example0.cxx
 *
 *
 */

#include <comprexchange.hxx>

main(int argc, char* argv[])
{

  try {

    gaspi::Runtime runtime;

    gaspi::Context context;

    gaspi::segment::Segment segment(1024*1024);


    std::string myMessage("Hi");
    comprEx::MyClass myClass(myMessage);
    myClass.printMessage();


    return EXIT_SUCCESS;
  } // try
  catch(...) {
     return EXIT_FAILURE;
  }

  return 0;
}

