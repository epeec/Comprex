#add executable called "example0" that is built from the source file
# "example.cxx". The extensions are automatically found.
add_executable (main 
        main.cc)

add_library(custom_protobufs STATIC
        DNN.pb.cc
        Layer.pb.cc)
        
add_dependencies(custom_protobufs
        generate_DNN
        generate_Layer)

message(" crr src dir:  ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (main 
                    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
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
			   rt
			   custom_protobufs)

            
add_custom_command(OUTPUT DNN.pb.cc
            COMMAND protoc ARGS -I=${CMAKE_CURRENT_SOURCE_DIR}/layers/ --cpp_out=. ${CMAKE_CURRENT_SOURCE_DIR}/layers/DNN.proto)

add_custom_command(OUTPUT Layer.pb.cc
            COMMAND protoc ARGS -I=${CMAKE_CURRENT_SOURCE_DIR}/layers/ --cpp_out=. ${CMAKE_CURRENT_SOURCE_DIR}/layers/Layer.proto)  
            
add_custom_target(generate_DNN
            DEPENDS DNN.pb.cc)
            
add_custom_target(generate_Layer
            DEPENDS Layer.pb.cc)
            
set_source_files_properties(DNN.pb.cc
            PROPERTIES GENERATED TRUE)

set_source_files_properties(Layer.pb.cc
            PROPERTIES GENERATED TRUE)
            

add_custom_target(run 
            ${CMAKE_BINARY_DIR}/src/main ${MNIST_PATH})
            
add_custom_target(gaspi_run 
            #echo "localhost\nlocalhos" > ${CMAKE_BINARY_DIR}/nodelist
            gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/src/main ${MNIST_PATH} ${CMAKE_BINARY_DIR}/checkpoints/ )
