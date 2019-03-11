/*
 *  example0.cxx
 *
 *
 */


#include <comprex.hxx>
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

    gaspi::segment::Segment segment(1024*1024);

    // first print some message
    std::string myMessage("Hi");
    compressed_exchange::MyClass myClass(myMessage);
    myClass.printMessage();

    // allocate3 a local vector and fill it in with some numbers
    const int localSize=10;
    std::unique_ptr<double[]> myVect = 
              std::unique_ptr<double[]> (new double [localSize]);

    // random generator(s) 
    random_generator::MyRandomGen<double> randmG(1, 1000);
    random_generator::MyRandomGen<double> randmG_int(1, 1000);

    //gaspi::group::Rank myRank =  context.rank();
    int myRnk = static_cast<int> ( context.rank().get() ); 
    for(int i = 0; i < localSize; i++) {
      myVect[i] =  randmG.generateRandomNumber();
        // localSize * static_cast<double> (context.rank().get()) + i;                
    }

    for(int i = 0; i < localSize; i++) {
      printf("\n [%d] myVect[%d]:%e ", myRnk, i, myVect[i]);
    }

    printf("\n\n After the exchange \n\n");
    
    // allocate src- and target- buffers on both ranks       
    gaspi::singlesided::write::SourceBuffer
         srcBuff(segment, localSize*sizeof(double)) ;
    gaspi::singlesided::write::TargetBuffer 
         targetBuff(segment, localSize*sizeof(double));

    int tag = 1;  // what is this for ??
    int tag1 =2;
    double* srcBuffEntry;
    double* targBuffEntry;

    gaspi::group::Rank neighbRank(0);
    if(context.rank() == gaspi::group::Rank(0)) {
      neighbRank++;
    }

    //-----
    compressed_exchange::ComprEx<double> cmprex(runtime, context, segment);
    cmprex.Compress_and_WriteRemote(myVect, 10, 0.2, neighbRank);
    //cmprex.Compress_and_WriteRemote(std::move(myVect), 10, 0.2, neighbRank);

   
    compressed_exchange::ComprEx<int> cmprex_int(runtime, context, segment);    
    std::unique_ptr<int[]> myVect_int = 
              std::unique_ptr<int[]> (new int [localSize]);
    for(int i = 0; i < localSize; i++) {
      myVect_int[i] =  randmG_int.generateRandomNumber();
        // localSize * static_cast<double> (context.rank().get()) + i;                
    }
    for(int i = 0; i < localSize; i++) {
      printf("\n [%d] myVect_int[%d]:%d ", myRnk, i, myVect_int[i]);      
    }
    cmprex_int.Compress_and_WriteRemote(myVect_int, 10, 1, neighbRank);
    //-----

    // rank #0
    if(context.rank() == gaspi::group::Rank(0)) {

      printf("\n [%d] neigbRank:%d \n", myRnk, neighbRank.get());

      srcBuff.connectToRemoteTarget(context,
                                     neighbRank,
                                     tag).waitForCompletion();

      targetBuff.connectToRemoteSource(context,
                                     neighbRank,
                                     tag1).waitForCompletion();

      
       // now copy the data to src buffer
       srcBuffEntry = (reinterpret_cast<double *>(srcBuff.address()));
       for(int i = 0; i < localSize; i++) {
	 srcBuffEntry[i] = myVect[i];
       }

       for(int i = 0; i < localSize; i++) {
         printf("\n [%d] srcBuffEntry[%d]:%e ", myRnk, i, srcBuffEntry[i]);
       }
       

       srcBuff.initTransfer(context);
       targetBuff.waitForCompletion(); 

       // print the target buffer
       targBuffEntry = (reinterpret_cast<double *>(targetBuff.address()));
       for(int i = 0; i < localSize; i++) {
	 myVect[i] = targBuffEntry[i];
       }
      for(int i = 0; i < localSize; i++) {
        printf("\n [%d] after exch myVect[%d]:%e ", myRnk, i, targBuffEntry[i]); //myVect[i]);
      }
       
    } // rank#0

    // perform the communication, rank #1
    if(context.rank() == gaspi::group::Rank(1)) {

      printf("\n [%d] neigbRank:%d \n", myRnk, neighbRank.get());


      targetBuff.connectToRemoteSource(context,
                                        neighbRank,
                                        tag).waitForCompletion();
      
      srcBuff.connectToRemoteTarget(context,
                                     neighbRank,
                                     tag1).waitForCompletion();

       // now copy the data to src buffer
       srcBuffEntry = (reinterpret_cast<double *>(srcBuff.address()));
       for(int i = 0; i < localSize; i++) {
          srcBuffEntry[i] = myVect[i];
       }
      
       
       srcBuff.initTransfer(context);
       targetBuff.waitForCompletion(); 
       
       // print the target buffer
       targBuffEntry = (reinterpret_cast<double *>(targetBuff.address()));
       for(int i = 0; i < localSize; i++) {
	 myVect[i] = targBuffEntry[i];
       }
      for(int i = 0; i < localSize; i++) {
        printf("\n [%d] after exch myVect[%d]:%e ", myRnk, i, targBuffEntry[i]); //myVect[i]);
      }
       

    }

    return EXIT_SUCCESS;
  } // try
  catch(...) {
     return EXIT_FAILURE;
  }

  return 0;
}

