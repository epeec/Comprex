#add executable called "example0" that is built from the source file
# "example.cxx". The extensions are automatically found.
add_executable (main 
        main.cc
        DNN.pb.cc)

message(" crr src dir:  ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (main 
                    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/layers
                    PRIVATE ${COMPREX_ROOT}/src)

# Link the executable to the Hello library. Since the Hello library has
# public include directories we will use those link directories when building
# helloDemo

target_link_libraries (main
               LINK_PUBLIC "-static-libstdc++"
               ComprEx 
			   GaspiCxx
			   ${GPI2_LIBRARIES} 
			   pthread 
			   protobuf
               ibverbs
			   rt)
			   
# add_custom_target(run 
#            ${CMAKE_BINARY_DIR}/src/main ${MNIST_PATH})
            
add_custom_target(gaspi_run 
            #echo "localhost\nlocalhos" > ${CMAKE_BINARY_DIR}/nodelist
            gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/src/main ${MNIST_PATH} ${CMAKE_BINARY_DIR}/checkpoints/ )
