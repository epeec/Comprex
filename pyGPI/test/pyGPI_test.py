import sys
sys.path.append('/u/l/loroch/PROGRAMMING/comprex/')
import pyGPI
from pyGPI import gaspi_printf
import time

#print("Start")

# to see messages from gaspi_printf, start Logger in your console with 'gaspi_logger&'
gaspi_printf("Message from GASPI Logger, before runtime. Start 'gaspi_logger&'!")

gaspi_runtime = pyGPI.Gaspi_Runtime()
gaspi_context = pyGPI.Gaspi_Context()
gaspi_segment = pyGPI.Gaspi_Segment(10)

myRank = gaspi_context.rank()

#with gpi.Gaspi_Runtime(), gpi.Gaspi_Context() as gaspi_context, gpi.Gaspi_Segment(10):
#print( "Runtime available: %r"%gpi.isRuntimeAvailable() )
#print("My Rank is %d"%(gaspi_context.getRank()) )

# test for barrier
gaspi_context.barrier()

for i in range(10):
    if myRank==0:
        time.sleep(1)
    gaspi_context.barrier()
    gaspi_printf("Rank %02d synchronized in iteration %02d"%(myRank, i))