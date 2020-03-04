import sys
sys.path.append('..')
import Gpi
from Gpi import gaspi_printf
import time

# to see messages from gaspi_printf, start Logger in your console with 'gaspi_logger&'
print("Start 'gaspi_logger&' in your console to see debugging output!")

# set up Gaspi environment
gaspi_runtime = Gpi.Gaspi_Runtime()
gaspi_context = Gpi.Gaspi_Context()
gaspi_segment = Gpi.Gaspi_Segment(10)

# get local rank
myRank = gaspi_context.rank()

# check if runtime initialized correctly
gaspi_printf( "Gaspi runtime available: %r"%Gpi.isRuntimeAvailable() )

# test for barrier
if myRank==0:
    gaspi_printf("The following messages should appear in 1 sec intervalls.")
for i in range(10):
    if myRank==0:
        time.sleep(1)
    gaspi_context.barrier()
    gaspi_printf("Rank %02d synchronized in iteration %02d"%(myRank, i))

if myRank==0:
    print("+++PASSED+++")