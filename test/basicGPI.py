import sys
sys.path.append('..')
import gpi
from gpi import gaspi_printf
import time

#print("Start")

# to see messages from gaspi_printf, start Logger in your console with 'gaspi_logger&'
gaspi_printf("Message from GASPI Logger, before runtime. Start 'gaspi_logger&'!")

gpi.initGaspiCxx()

gaspi_runtime = gpi.Gaspi_Runtime()
gaspi_segment = gpi.Gaspi_Segment(10)

myRank = gaspi_runtime.getGlobalRank()

# test for barrier
gaspi_runtime.barrier()

for i in range(10):
    if myRank==0:
        time.sleep(1)
    gaspi_runtime.barrier()
    gaspi_printf("Rank %02d synchronized in iteration %02d"%(myRank, i))