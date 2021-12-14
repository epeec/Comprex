# Example gaspiEx
add_executable (example_gaspiEx_c gaspiEx.cpp)
target_include_directories(example_gaspiEx_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (example_gaspiEx_c
                LINK_PUBLIC "-static-libstdc++"
			   	GaspiCxx
			    GPI2::GPI2
			   	pthread
			   	rt)
add_custom_target(run_example_gaspiEx_c
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example_gaspiEx_c)

# Example comprEx
add_executable (example_comprEx_c comprEx.cpp)
target_include_directories(example_comprEx_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (example_comprEx_c
                LINK_PUBLIC "-static-libstdc++"
			   	GaspiCxx
			   	GPI2::GPI2
			   	pthread
			   	rt)
add_custom_target(run_example_comprEx_c
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example_comprEx_c)

# Example allreduce_speed
add_executable (example_allreduce_speed_c allreduce_speed.cpp)
target_include_directories(example_allreduce_speed_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (example_allreduce_speed_c
                LINK_PUBLIC "-static-libstdc++"
			   	GaspiCxx
			   	GPI2::GPI2
            	)
add_custom_target(run_example_allreduce_speed_c
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example_allreduce_speed_c)

# Example comprex_speed
add_executable( example_comprex_speed_c comprex_speed.cpp )
target_include_directories (example_comprex_speed_c PRIVATE ${COMPREX_SRC})
target_link_libraries (example_comprex_speed_c
                LINK_PUBLIC "-static-libstdc++"
			    GaspiCxx
			    GPI2::GPI2
			    pthread
                rt)
add_custom_target(run_example_comprex_speed_c
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example_comprex_speed_c)

# Example Python gaspiEx
set (TESTNAME example_gaspiEx_py) 
file(WRITE ${COMPREX_BUILD}/${TESTNAME}.sh "#!/bin/bash -i\nconda activate ${CONDA_ENVIRONMENT}\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/examples/gaspiEx.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/${TESTNAME}.sh )
add_custom_target(run_${TESTNAME}
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/${TESTNAME}.sh)

# Example Python comprEx
set (TESTNAME example_comprEx_py) 
file(WRITE ${COMPREX_BUILD}/${TESTNAME}.sh "#!/bin/bash -i\nconda activate ${CONDA_ENVIRONMENT}\ncd ${CMAKE_SOURCE_DIR}/build\n${PYTHON} ${COMPREX_ROOT}/examples/comprEx.py")
execute_process( COMMAND chmod +x ${COMPREX_BUILD}/${TESTNAME}.sh )
add_custom_target(run_${TESTNAME}
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/${TESTNAME}.sh)