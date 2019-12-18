
import pyGPI.Gpi as gpi

# gaspi setup
global gaspi_runtime
gaspi_runtime = gpi.Gaspi_Runtime()
global gaspi_context
gaspi_context= gpi.Gaspi_Context()
global gaspi_segment
gaspi_segment= gpi.Gaspi_Segment(2**27) # 128 MB