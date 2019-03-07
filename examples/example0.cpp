/*
 *  example0.cxx
 *
 *
 */



#include <comprexchange.hxx>

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
    comprEx::MyClass myClass(myMessage);
    myClass.printMessage();

    // allocate3 a local vector and fill it in with some numbers
    const int localSize=10;
    std::unique_ptr<double[]> myVect = 
              std::unique_ptr<double[]> (new double [localSize]);

    //gaspi::group::Rank myRank =  context.rank();
    int myRnk = static_cast<int> ( context.rank().get() ); 
    for(int i = 0; i < localSize; i++) {
      myVect[i] = localSize * static_cast<double> (context.rank().get()) + i;   
    }

    for(int i = 0; i < localSize; i++) {
      printf("\n [%d] myVect[%d]:%e ", myRnk, i, myVect[i]);
    }

    printf("\n\n After the exchange \n\n");

    gaspi::group::Rank srcRank(gaspi::group::Rank(0));
    gaspi::group::Rank destRank(gaspi::group::Rank(1));
    
    // allocate src- and target- buffers, rnak#0 -> src, rank#1 -> target       
    // perform the communication, rank#0 sends 
    // (i.e., initTransfer on the source buff)
    int tag = 1;
    if(context.rank() == gaspi::group::Rank(0)) {
       gaspi::singlesided::write::SourceBuffer 
	 srcBuff(segment, localSize*sizeof(double));       

       srcBuff.connectToRemoteTarget(context,
                                     destRank,
                                     tag).waitForCompletion();

       // now copy the data to src buffer
       double* srcBuffEntry = (reinterpret_cast<double *>(srcBuff.address()));
       for(int i = 0; i < localSize; i++) {
	 srcBuffEntry[i] = myVect[i];
       }

       for(int i = 0; i < localSize; i++) {
         printf("\n [%d] srcBuffEntry[%d]:%e ", myRnk, i, srcBuffEntry[i]);
       }
         
       srcBuff.initTransfer(context);
       
    }

    // perform the communication, rank#1 receives 
    // (i.e., waitForCompletion on targetBuff) 
    if(context.rank() == gaspi::group::Rank(1)) {
       gaspi::singlesided::write::TargetBuffer 
         targetBuff(segment, localSize*sizeof(double));

       targetBuff.connectToRemoteSource(context,
                                        srcRank,
                                        tag).waitForCompletion();
       targetBuff.waitForCompletion(); 

       // print the target buffer
       double* targBuffEntry = (reinterpret_cast<double *>(targetBuff.address()));
       for(int i = 0; i < localSize; i++) {
	 myVect[i] = targBuffEntry[i];
       }
      for(int i = 0; i < localSize; i++) {
        printf("\n [%d] myVect[%d]:%e ", myRnk, i, targBuffEntry[i]); //myVect[i]);
      }


    }



    return EXIT_SUCCESS;
  } // try
  catch(...) {
     return EXIT_FAILURE;
  }

  return 0;
}

