#Create the ComprEx library.
# The extension is already found. Any number of sources could be listed here.
add_executable( threshold_test threshold_test.cpp )
add_executable( compressor_test compressor_test.cpp )

message(" crr src dir:  ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (threshold_test PRIVATE ${COMPREX_SRC})
target_include_directories (compressor_test PRIVATE ${COMPREX_SRC})


add_custom_target(gaspi_pyGPI_test
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/gaspi_pyGPI_test.sh)


file(WRITE ${COMPREX_BUILD}/gaspi_pyGPI_test.sh "#!/bin/bash\nexport PYTHONPATH=$ENV{PYTHONPATH}:${COMPREX_ROOT}/python\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/test/pyGPI_test.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/gaspi_pyGPI_test.sh )