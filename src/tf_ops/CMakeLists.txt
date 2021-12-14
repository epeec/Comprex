#Create .so file for Tensorflow Ops.

add_library ( tfGPIOps SHARED
    tfAllreduce.cc
 )

target_link_libraries(tfGPIOps
        Tensorflow::Tensorflow
		GPI2::GPI2
        )

install(TARGETS tfGPIOps EXPORT
	ARCHIVE  DESTINATION ${CMAKE_INSTALL_DIR}/lib
	LIBRARY  DESTINATION ${CMAKE_INSTALL_DIR}/lib)

