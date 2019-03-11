/*
 *  example0.cxx
 *
 *
 */

#include <random>

#include <comprex.hxx>


namespace random_generator {

  template <class VarTYPE>
  class MyRandomGen {
    private:
     VarTYPE _lowerBound;
     VarTYPE _upperBound;


    //std::unique_ptr<std::uniform_real_distribution<double> > _unifDistr;
    // std::unique_ptr<std::default_random_engine > _randmEngine;

     std::random_device _randmDevice;
     std::unique_ptr<std::mt19937> _engine; // standard mersenne twister engine
     std::unique_ptr< std::uniform_real_distribution<VarTYPE> > _realDistr;
     std::unique_ptr< std::uniform_int_distribution<>  > _intDistr;

    public:

     MyRandomGen(VarTYPE lowerBnd, VarTYPE upperBnd)
       :
         _lowerBound(lowerBnd)
       , _upperBound(upperBnd)
	 , _randmDevice()
         , _engine()
         , _realDistr()
     {

       _engine =  std::unique_ptr<std::mt19937>
         (new std::mt19937(_randmDevice()));  
       _realDistr = std::unique_ptr< std::uniform_real_distribution<VarTYPE> >
	 (new std::uniform_real_distribution<VarTYPE> (_lowerBound, _upperBound) );
       _intDistr = std::unique_ptr< std::uniform_int_distribution<> >
        (new std::uniform_int_distribution<> (
					      static_cast<int>(_lowerBound), static_cast<int>(_upperBound)) );

     }

    // specialize it for ints later on
     double generateRandomNumber() 
     {
       //std::random_device rd;  //Will be used to obtain a seed for the random number engine
       //std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
       //std::uniform_real_distribution<> dis(1.0, 2.0);
        // Use dis to transform the random unsigned int generated by gen into a 
        // double in [1, 2). Each call to dis(gen) generates a new random double
       //std::cout << dis(gen) << ' ';

        return (*_realDistr)(*_engine);
     }
      

  };// MyRandomGen 

} // namespace random generator

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

    // random generator 
    random_generator::MyRandomGen<double> randmG(1, 1000);

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

