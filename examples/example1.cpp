/*
 *  example1.cxx
 *
 *
 */


#include <comprex.hxx>
#include <myRandomGen.cxx>

main(int argc, char* argv[])
{

  try {

    // init GaspiCxx
    gaspi::Runtime runtime;
    gaspi::Context context;

    if(context.size().get() != 2 ) {
      printf("\n To be run on two ranks only !!\n");
      throw;
    }

    if(argc != 2) {
       printf("Wrong number of command line arguments !! \n");
       printf("Usage: \n");
       printf("./EXAMPLE <nThreads> \n");
       return -1;
    }

    int nThreads =  atoi(argv[1]);

    gaspi::segment::Segment segment( static_cast<size_t>(1) << 28);


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
    int myTopK = 43;  // communicatze 43% of the top-values

    // allocate local vector (send-buffer on rank#0, recv-buff on rank #1) 
    std::unique_ptr<int[]> myVect_int = 
              std::unique_ptr<int[]> (new int [localSize]);
    //for(int i = 0; i < localSize; i++) {
    //  myVect_int[i] =  randmG_int.generateRandomNumber();      
    //}


    // thread´s pining
    const int nCores = 8; // seislab, sandy-Bridge, NUMA-aware GPI
    std::unique_ptr<int[]> pinPattern = 
              std::unique_ptr<int[]> (new int [nCores]);    
    if(nCores < nThreads) {  // can not pin the threads !!!
       throw std::runtime_error (
            " Number of threads per rank higher than the number of cores..");
    }
    if (static_cast<int>(myRank.get()) % 2 == 0) 
          for(int i = 0; i < nCores; i++) pinPattern[i] = i*2;
    if (static_cast<int>(myRank.get()) % 2 == 1) 
          for(int i = 0; i < nCores; i++) pinPattern[i] = i*2+1;

    // with pin-pattern, pin the threads     
    compressed_exchange::ComprExTopK<int>
      cmprex_int(runtime, context, segment, localSize, 
                                  nThreads, pinPattern.get());
    
    // without pin-pattern, do not pin, i.e. pinPattern = NULL by default 
    //compressed_exchange::ComprExTopK<int>
    //  cmprex_int(runtime, context, segment, localSize, nThreads);
    
    // single-threaded by default, also no pin-pattern
    //compressed_exchange::ComprExTopK<int>
    //  cmprex_int(runtime, context, segment, localSize); 
      
    const int * pRestsV = cmprex_int.entryPointerRestsVector();    

    const int nSweeps = 3; //3; 
for(int ii = 0; ii < nSweeps; ii++) {  

  context.barrier();
  printf("\n  [%d] ==================== Attempt No:%d \n", myRnk_int, ii);

    if(myRank == srcRank) {

        int factr = 1;
        for(int i = 0; i < localSize; i++) {
         if(i%2 == 1) factr = -1;
         else factr = 1; 
         myVect_int[i] =  factr * randmG_int.generateRandomNumber();
          // localSize * static_cast<double> (context.rank().get()) + i;  
        }

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

    if(myRank == destRank) {
        for(int i = 0; i < localSize; i++) {
	  myVect_int[i] = 0;
        }
    }

context.barrier();

    if(myRank == srcRank) {
      cmprex_int.compress_and_p2pVectorWriteRemote(
      	  myVect_int, myTopK, destRank, tag);

   
//cmprex_int.printAuxiliaryInfoVector("/scratch/stoyanov/comprEx/run");
    } //myRank == srcRank  

    
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
cmprex_int.flushTheRestsVector();

//cmprex_int.printAuxiliaryInfoVector("/scratch/stoyanov/comprEx/run");


    if(myRank == srcRank) {
        printf("\n"); 
        for(int i = 0; i < localSize; i++) {
        printf("\n [%d] outer_loops over, after flushing the rests, restsVector[%d]:%d ", 
	       myRnk_int, i, pRestsV[i]);      
        }
        printf("\n"); 
    }//if(myRank == srcRank)

    // cmprex_int.printCompressedVector("/scratch/stoyanov/comprEx/run");
    //cmprex_int.printAuxiliaryInfoVector("/scratch/stoyanov/comprEx/run");
    //cmprex_int.printCompressedVector_inOriginalSize(
    //                                 "/scratch/stoyanov/comprEx/run");

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

