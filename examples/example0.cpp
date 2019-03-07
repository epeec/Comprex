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

    // rank #0
    if(context.rank() == gaspi::group::Rank(0)) {

      //gaspi::group::Rank neighbRank(1);

      srcBuff.connectToRemoteTarget(context,
                                     neighbRank,
                                     tag).waitForCompletion();

      //targetBuff.connectToRemoteSource(context,
      //                               neighbRank,
      //                               tag1).waitForCompletion();

       // now copy the data to src buffer
       srcBuffEntry = (reinterpret_cast<double *>(srcBuff.address()));
       for(int i = 0; i < localSize; i++) {
	 srcBuffEntry[i] = myVect[i];
       }

       for(int i = 0; i < localSize; i++) {
         printf("\n [%d] srcBuffEntry[%d]:%e ", myRnk, i, srcBuffEntry[i]);
       }
         
       srcBuff.initTransfer(context);
       //targetBuff.waitForCompletion(); 

       // print the target buffer
       /*
       targBuffEntry = (reinterpret_cast<double *>(targetBuff.address()));
       for(int i = 0; i < localSize; i++) {
	 myVect[i] = targBuffEntry[i];
       }
      for(int i = 0; i < localSize; i++) {
        printf("\n [%d] myVect[%d]:%e ", myRnk, i, targBuffEntry[i]); //myVect[i]);
      }
       */
       
    }

    // perform the communication, rank #1
    if(context.rank() == gaspi::group::Rank(1)) {

      //gaspi::group::Rank neighbRank(0);

      //srcBuff.connectToRemoteTarget(context,
      //                               neighbRank,
      //                               tag1).waitForCompletion();

       targetBuff.connectToRemoteSource(context,
                                        neighbRank,
                                        tag).waitForCompletion();


       // now copy the data to src buffer
       //srcBuffEntry = (reinterpret_cast<double *>(srcBuff.address()));
       //for(int i = 0; i < localSize; i++) {
       // srcBuffEntry[i] = myVect[i];
       //}

       
       //srcBuff.initTransfer(context);
       targetBuff.waitForCompletion(); 

       // print the target buffer
       targBuffEntry = (reinterpret_cast<double *>(targetBuff.address()));
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

