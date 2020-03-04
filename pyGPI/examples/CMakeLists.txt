#add executable called "example1" that is built from the source file
add_executable (example1 example1.cpp)

message(" crr src dir:  ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories (example1 PRIVATE ${COMPREX_SRC})

target_link_libraries (example1
                LINK_PUBLIC "-static-libstdc++"
			   	GaspiCxx
			   	${GPI2_LIBRARIES}
			   	pthread
            	ibverbs
			   	rt)

add_custom_target(gaspi_example1_cpp
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example1)

add_custom_target(gaspi_example1_py
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/gaspi_example1_py.sh)
			   

# create gaspi_run scripts for python
file(WRITE ${COMPREX_BUILD}/gaspi_example1_py.sh "#!/bin/bash\nexport PYTHONPATH=$ENV{PYTHONPATH}:${COMPREX_ROOT}\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/examples/example1.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/gaspi_example1_py.sh )
