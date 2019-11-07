#add executable called "example0" that is built from the source file
# "example.cxx". The extensions are automatically found.
add_executable (example0 example0.cpp)

message(" crr src dir:  ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (example0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link the executable to the Hello library. Since the Hello library has
# public include directories we will use those link directories when building
# helloDemo

target_include_directories (example0 PRIVATE ${COMPREX_SRC})

target_link_libraries (example0
                LINK_PUBLIC "-static-libstdc++"
			   	GaspiCxx
			   	${GPI2_LIBRARIES}
			   	pthread
            	ibverbs
			   	rt)

add_custom_target(gaspi_example0_cpp
               #echo "localhost\nlocalhost" > ${CMAKE_BINARY_DIR}/nodelist
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example0)

add_custom_target(gaspi_example1_cpp
               #echo "localhost\nlocalhost" > ${CMAKE_BINARY_DIR}/nodelist
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/examples/example1)

add_custom_target(gaspi_example0_py
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/gaspi_example0_py.sh)
			   
add_custom_target(gaspi_allreduce_py
			   gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/gaspi_allreduce_py.sh)
