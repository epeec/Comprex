/*
 *  example0.cxx: a "minimal" example to test the ComprEx library.
 *
 *  Short description:
 *   - the example works on two ranks: source (#0) and destination (#1) rank;
 *   - first, GaspiCxx-objects like Runtime, Context, Segment, and Rank 
 *           are instantiated;
 *   - on each ranks an integer array is allocated and filled-in with   
 *         randomly generated numbers within the interval [1, 1000];
 *   - the ComprEx-class ComprExRunLengths is then instantiated by
 *       compressed_exchange::ComprExRunLengths<int>
 *             cmprex_int(runtime, context, segment, localSize);
 *      
 *      This definition says that integer vectors of size #localSize can
 *      be exchanged using the Run-Lenths-Encoding as compression method
 *   
 *   - the, for a threshold_value==400, 
 *     -- the source rank calls 
 *        cmprex_int.compress_and_p2pVectorWriteRemote(..),
 *         (i.e. the original vector is to be first compressed and then sent) 
 *     -- while the destination ranks correspondingly calls
 *        cmprex_int.p2pVectorGetRemote(..)
 *         (i.e. gets the compressed vector from the source rank and 
 *                                         unpacks it)
 *
 *  
 */


#include <comprex.hxx>
#include <myRandomGen.cxx>

main(int argc, char* argv[])
{

  try {

    // init GaspiCxx: instantiate Runtime, Context, and Segment
    gaspi::Runtime runtime;
    gaspi::Context context;

    if(context.size().get() != 2 ) {
      printf("\n To be run on two ranks only !!\n");
      throw;
    }

    gaspi::segment::Segment segment( static_cast<size_t>(1) << 28);


    // allocate3 a local vector and fill it in with some numbers
    const int localSize=10;
    // allocate a double vector and fill it in 
    //std::unique_ptr<double[]> myVect = 
    //          std::unique_ptr<double[]> (new double [localSize]);

    // instantiate a random generator, generating doubles
    random_generator::MyRandomGen<double> randmG_int(1, 1000);


    // define the ranks, rank#0: source rank, ramk#1:destination rank    
    gaspi::group::Rank myRank =  context.rank();
    int myRnk_int = static_cast<int> ( context.rank().get() ); 
    //gaspi::group::Rank neighbRank(0);
    //if(context.rank() == gaspi::group::Rank(0)) {
    //  neighbRank++;
    //}

    gaspi::group::Rank srcRank(0);
    gaspi::group::Rank destRank(1);    
    int tag = 1;  // communivation  tag
    int myTreshold = 400;  // threshold  

    // allocate local vector (send-buffer on rank#0, recv-buff on rank #1) 
    std::unique_ptr<int[]> myVect_int = 
              std::unique_ptr<int[]> (new int [localSize]);
    for(int i = 0; i < localSize; i++) {
      myVect_int[i] =  randmG_int.generateRandomNumber();
      // localSize * static_cast<double> (context.rank().get()) + i;
    }

    // Intsantiate the Run-Length-Encoding class of ComprEx 
    compressed_exchange::ComprExRunLengths<int>
      cmprex_int(runtime, context, segment, localSize);

    // get an "entry pointer" to access the so called rests-vector
    const int * pRestsV = cmprex_int.entryPointerRestsVector();

    // threads and pining, just to show how it look like    
    const int nCores = 8; 
    std::unique_ptr<int[]> pinPattern = 
              std::unique_ptr<int[]> (new int [nCores]);    
    for(int i = 0; i < nCores; i++) { // sandy-bridge,..
      if(i % 2 == 0) pinPattern[i] = i*2;
      if(i % 2 == 1) pinPattern[i] = i;
    }
    cmprex_int.setPinPatternForTheThreads(pinPattern);

 
    // start three compress-and-exchange sweeps between src and dest rank
for(int ii = 0; ii < 3; ii++) { 

  context.barrier();
  printf("\n  [%d] ==================== Attempt No:%d \n", myRnk_int, ii);

    // fill-in the contents of the local vector
    for(int i = 0; i < localSize; i++) {
      myVect_int[i] =  randmG_int.generateRandomNumber();
        // localSize * static_cast<double> (context.rank().get()) + i;                
    }

    // print out the current contents of the local vector
    if(myRank == srcRank) {

        printf("\n"); 
        for(int i = 0; i < localSize; i++) {
        printf("\n [%d] outer_loop:%d, before compress & exchange, myVect_int[%d]:%d ", 
	       myRnk_int, ii, i, myVect_int[i]);      
        }
        printf("\n"); 

        printf("\n"); 
        for(int i = 0; i < localSize; i++) {
        printf("\n [%d] outer_loop:%d, before compress & exchange, restsVector[%d]:%d ", 
	       myRnk_int, ii, i, pRestsV[i]);      
        }
        printf("\n"); 

    }//if(myRank == srcRank)

context.barrier();

    // the source rank compresses and sends
    if(myRank == srcRank) {
      cmprex_int.compress_and_p2pVectorWriteRemote(
		  myVect_int, myTreshold, destRank, tag);
    } 

    // the destination rank gets the vector from the source rank 
    // and uncompresses it
    if(myRank == destRank) {
      cmprex_int.p2pVectorGetRemote(
		  myVect_int, srcRank, tag);
      
      printf("\n"); 
      for(int i = 0; i < localSize; i++) {
          printf("\n [%d] outer_loop:%d, after exchange, myVect_int[%d]:%d ", 
		 myRnk_int, ii, i, myVect_int[i]);      
      }//for
      printf("\n"); 
    }// if(myRank == destRank)

     
    if(myRank == srcRank) {
        printf("\n"); 
        for(int i = 0; i < localSize; i++) {
        printf("\n [%d] outer_loop:%d, after compress & exchange, restsVector[%d]:%d ", 
	       myRnk_int, ii, i, pRestsV[i]);      
        }
        printf("\n"); 

    }//if(myRank == srcRank)

  context.barrier();

 } //ii

// flush the rests-vector on the source rank 
cmprex_int.flushTheRestsVector();

    if(myRank == srcRank) {
        printf("\n"); 
        for(int i = 0; i < localSize; i++) {
        printf("\n [%d] outer_loops over, after flushing the rests, restsVector[%d]:%d ", 
	       myRnk_int, i, pRestsV[i]);      
        }
        printf("\n"); 
    }//if(myRank == srcRank)

    cmprex_int.printCompressedVector("/scratch/stoyanov/comprEx/run");
    cmprex_int.printAuxiliaryInfoVector("/scratch/stoyanov/comprEx/run");
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

