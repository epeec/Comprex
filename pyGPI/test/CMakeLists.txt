#Create the ComprEx library.
# The extension is already found. Any number of sources could be listed here.
add_executable( threshold_test threshold_test.cpp )
add_executable( compressor_test compressor_test.cpp )
add_executable( compressedExchange_test compressedExchange_test.cpp )
add_executable( gaspiExchange_test gaspiExchange_test.cpp )

message(" crr src dir:  ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (threshold_test PRIVATE ${COMPREX_SRC})
target_include_directories (compressor_test PRIVATE ${COMPREX_SRC})
target_include_directories (compressedExchange_test PRIVATE ${COMPREX_SRC})
target_include_directories (gaspiExchange_test PRIVATE ${COMPREX_SRC})

target_link_libraries (compressedExchange_test
               LINK_PUBLIC "-static-libstdc++"
			   GaspiCxx
			   ${GPI2_LIBRARIES}
			   pthread
               ibverbs
               rt)

target_link_libraries (gaspiExchange_test
               LINK_PUBLIC "-static-libstdc++"
			   GaspiCxx
			   ${GPI2_LIBRARIES}
			   pthread
               ibverbs
			   rt)

add_custom_target(gaspi_compressedExchange_test
               #echo "localhost\nlocalhos" > ${CMAKE_BINARY_DIR}/nodelist
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/test/compressedExchange_test)

add_custom_target(gaspi_gaspiExchange_test
               #echo "localhost\nlocalhos" > ${CMAKE_BINARY_DIR}/nodelist
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${CMAKE_BINARY_DIR}/test/gaspiExchange_test)

add_custom_target(gaspi_pyGPI_test
               #echo "localhost\nlocalhos" > ${CMAKE_BINARY_DIR}/nodelist
               gaspi_run -m ${CMAKE_BINARY_DIR}/nodelist ${COMPREX_BUILD}/gaspi_pyGPI_test.sh)