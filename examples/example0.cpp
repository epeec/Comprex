/*
 *  example0.cxx
 *
 *
 */


//#include <comprex.hxx>
#include <runLenCompressr.hxx>
#include <myRandomGen.cxx>

main(int argc, char* argv[])
{

  try {

    // init GPI-2 
    gaspi::Runtime runtime;
    gaspi::Context context;

    if(context.size().get() != 2 ) {
      printf("\n To be run on two ranks only !!\n");
      throw;
    }

    gaspi::segment::Segment segment( static_cast<size_t>(1) << 28);

    // first print some message
    std::string myMessage("Hi");
    compressed_exchange::MyClass myClass(myMessage);
    myClass.printMessage();

    // allocate3 a local doubles-vector and fill it in with some numbers
    const int localSize=10;
    //std::unique_ptr<double[]> myVect = 
    //          std::unique_ptr<double[]> (new double [localSize]);

    // random generator(s) 
    //random_generator::MyRandomGen<double> randmG(1, 1000);
    random_generator::MyRandomGen<double> randmG_int(1, 1000);

    gaspi::group::Rank myRank =  context.rank();
    int myRnk_int = static_cast<int> ( context.rank().get() ); 
    //for(int i = 0; i < localSize; i++) {
    //  myVect[i] =  randmG.generateRandomNumber();
      // localSize * static_cast<double> (context.rank().get()) + i;
    //}

    //for(int i = 0; i < localSize; i++) {
    //  printf("\n [%d]before exchange,  myVect[%d]:%e ", myRnk_int, i, myVect[i]);
    //}
    //gaspi::group::Rank neighbRank(0);
    //if(context.rank() == gaspi::group::Rank(0)) {
    //  neighbRank++;
    //}


    gaspi::group::Rank srcRank(0);
    gaspi::group::Rank destRank(1);    
    int tag = 1;  // comm tag
    int myTreshold = 400;

    // allocate local vector (send-buffer on rank#0, recv-buff on rank #1) 
    std::unique_ptr<int[]> myVect_int = 
              std::unique_ptr<int[]> (new int [localSize]);
    for(int i = 0; i < localSize; i++) {
      myVect_int[i] =  randmG_int.generateRandomNumber();
        // localSize * static_cast<double> (context.rank().get()) + i;                
    }

    printf("\n"); 
    for(int i = 0; i < localSize; i++) {
      printf("\n [%d] before exchange, myVect_int[%d]:%d ", 
                                       myRnk_int, i, myVect_int[i]);      
    }
    printf("\n"); 

    compressed_exchange::impl::CompressorRunLengths<int>
            cmprex_int(runtime, context, segment);


    if(myRank == srcRank) {
      cmprex_int.compress_and_p2pVectorWriteRemote(
		  myVect_int, localSize, myTreshold, destRank, tag);
    } 

    if(myRank == destRank) {
      cmprex_int.p2pVectorGetRemote(
		  myVect_int, localSize, srcRank, tag);
      printf("\n"); 
      for(int i = 0; i < localSize; i++) {
          printf("\n [%d] after exchange, myVect_int[%d]:%d ", 
                                       myRnk_int, i, myVect_int[i]);      
      }//for
      printf("\n"); 
    }// if(myRank == destRank)

    cmprex_int.printCompressedVector("/scratch/stoyanov/comprEx/run");
    cmprex_int.printRunLengthsVector("/scratch/stoyanov/comprEx/run");
    cmprex_int.printCompressedVector_inOriginalSize(
                                     "/scratch/stoyanov/comprEx/run");
    return EXIT_SUCCESS;
  } // try
  catch (const std::exception &e) {
     std::cerr << e.what() << std::endl;
     return EXIT_FAILURE;
  }
  catch(...) {
     return EXIT_FAILURE;
  }

  return 0;
}

