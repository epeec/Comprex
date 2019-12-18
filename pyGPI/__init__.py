import pyGPI.Gpi as gpi

# gaspi Runtime
# there is only one runtime
global gaspi_runtime
gaspi_runtime = gpi.Gaspi_Runtime()

# gaspi context
# multiple contexts can be created
global gaspi_context
gaspi_context= gpi.Gaspi_Context()

def set_default_gaspi_context(context):
    del gaspi_context
    gaspi_context = context

# gaspi segment
# multiple segments can be created
global gaspi_segment
gaspi_segment= gpi.Gaspi_Segment(2**27) # 128 MB default size

def set_default_gaspi_segment(segment):
    del gaspi_segment
    gaspi_segment = segment