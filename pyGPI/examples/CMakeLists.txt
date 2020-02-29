#add executable called "example0" that is built from the source file
# "example.cxx". The extensions are automatically found.
add_executable (example0 example0.cpp)
add_executable (example1 example1.cpp)

message(" crr src dir:  ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (example0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link the executable to the Hello library. Since the Hello library has
# public include directories we will use those link directories when building
# helloDemo

target_include_directories (example0 PRIVATE ${COMPREX_SRC})
target_include_directories (example1 PRIVATE ${COMPREX_SRC})

target_link_libraries (example0
                LINK_PUBLIC "-static-libstdc++"
			   	GaspiCxx
			   	${GPI2_LIBRARIES}
			   	pthread
            	ibverbs
			   	rt)

target_link_libraries (example1
                LINK_PUBLIC "-static-libstdc++"
			   	GaspiCxx
			   	${GPI2_LIBRARIES}
			   	pthread
            	ibverbs
			   	rt)

add_custom_target(gaspi_example0_cpp
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example0)

add_custom_target(gaspi_example1_cpp
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example1)

add_custom_target(gaspi_example0_py
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/gaspi_example0_py.sh)

add_custom_target(gaspi_example1_py
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/gaspi_example1_py.sh)
			   
add_custom_target(gaspi_allreduce_py
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/gaspi_allreduce_py.sh)


# create gaspi_run script for python
file(WRITE ${COMPREX_BUILD}/gaspi_example0_py.sh "#!/bin/bash\nexport PYTHONPATH=$ENV{PYTHONPATH}:${COMPREX_ROOT}/python\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/examples/example0.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/gaspi_example0_py.sh )

file(WRITE ${COMPREX_BUILD}/gaspi_example1_py.sh "#!/bin/bash\nexport PYTHONPATH=$ENV{PYTHONPATH}:${COMPREX_ROOT}/python\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/examples/example1.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/gaspi_example1_py.sh )

file(WRITE ${COMPREX_BUILD}/gaspi_allreduce_py.sh "#!/bin/bash\nexport PYTHONPATH=$ENV{PYTHONPATH}:${COMPREX_ROOT}/python\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/examples/allreduce.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/gaspi_allreduce_py.sh )

file(WRITE ${COMPREX_BUILD}/gaspi_pyGPI_test.sh "#!/bin/bash\nexport PYTHONPATH=$ENV{PYTHONPATH}:${COMPREX_ROOT}/python\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/test/pyGPI_test.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/gaspi_pyGPI_test.sh )