# Test for the Threshold class.
add_executable( test_threshold_c threshold.cpp )
target_include_directories (test_threshold_c PRIVATE ${COMPREX_SRC})

# Test for the Compressor class.
add_executable( test_compressor_c compressor.cpp )
target_include_directories (test_compressor_c PRIVATE ${COMPREX_SRC})

# Test for the Comprex class.
add_executable( test_comprex_c comprex.cpp )
target_include_directories (test_comprex_c PRIVATE ${COMPREX_SRC})
target_link_libraries (test_comprex_c
               LINK_PUBLIC "-static-libstdc++"
			   GaspiCxx
			   GPI2::GPI2
			   pthread
               rt)
add_custom_target(run_test_comprex_c
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/test/test_comprex_c)

# Test AllToOneAllreduce
add_executable( test_AllToOneAllreduce_c AllToOneAllreduce.cpp )
target_include_directories (test_AllToOneAllreduce_c PRIVATE ${COMPREX_SRC})
target_link_libraries (test_AllToOneAllreduce_c
               LINK_PUBLIC "-static-libstdc++"
			   GaspiCxx
			   GPI2::GPI2
			   pthread
               rt)
add_custom_target(run_test_AllToOneAllreduce_c
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/test/test_AllToOneAllreduce_c)

# Test RingAllreduce
add_executable( test_RingAllreduce_c RingAllreduce.cpp )
target_include_directories (test_RingAllreduce_c PRIVATE ${COMPREX_SRC})
target_link_libraries (test_RingAllreduce_c
               LINK_PUBLIC "-static-libstdc++"
			   GaspiCxx
			   GPI2::GPI2
			   pthread
               rt)
add_custom_target(run_test_RingAllreduce_c
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/test/test_RingAllreduce_c)

# Test BigRingAllreduce
add_executable( test_BigRingAllreduce_c BigRingAllreduce.cpp )
target_include_directories (test_BigRingAllreduce_c PRIVATE ${COMPREX_SRC})
target_link_libraries (test_BigRingAllreduce_c
               LINK_PUBLIC "-static-libstdc++"
			   GaspiCxx
			   GPI2::GPI2
			   pthread
               rt)
add_custom_target(run_test_BigRingAllreduce_c
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/test/test_BigRingAllreduce_c)

# Test Tensorflow allreduceOp
set (TESTNAME test_TFallreduceOp_py)
file(WRITE ${COMPREX_BUILD}/${TESTNAME}.sh "#!/bin/bash -i\nconda activate ${CONDA_ENVIRONMENT}\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/test/TFallreduceOp.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/${TESTNAME}.sh )
add_custom_target(run_${TESTNAME}
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/${TESTNAME}.sh)

# Test Python AllToOneAllreduce
set (TESTNAME test_AllToOneAllreduce_py) 
file(WRITE ${COMPREX_BUILD}/${TESTNAME}.sh "#!/bin/bash -i\nconda activate ${CONDA_ENVIRONMENT}\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/test/AllToOneAllreduce.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/${TESTNAME}.sh )
add_custom_target(run_${TESTNAME}
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/${TESTNAME}.sh)

# Test Python RingAllreduce
set (TESTNAME test_RingAllreduce_py) 
file(WRITE ${COMPREX_BUILD}/${TESTNAME}.sh "#!/bin/bash -i\nconda activate ${CONDA_ENVIRONMENT}\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/test/RingAllreduce.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/${TESTNAME}.sh )
add_custom_target(run_${TESTNAME}
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/${TESTNAME}.sh)

# Test Python BigRingAllreduce
set (TESTNAME test_BigRingAllreduce_py) 
file(WRITE ${COMPREX_BUILD}/${TESTNAME}.sh "#!/bin/bash -i\nconda activate ${CONDA_ENVIRONMENT}\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/test/BigRingAllreduce.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/${TESTNAME}.sh )
add_custom_target(run_${TESTNAME}
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/${TESTNAME}.sh)

# Test Python basicGPI
set (TESTNAME test_basicGPI_py) 
file(WRITE ${COMPREX_BUILD}/${TESTNAME}.sh "#!/bin/bash -i\nconda activate ${CONDA_ENVIRONMENT}\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/test/basicGPI.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/${TESTNAME}.sh )
add_custom_target(run_${TESTNAME}
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/${TESTNAME}.sh)
