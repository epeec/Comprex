#Create the pyGPI library.

add_library ( PyGPI SHARED
    comprEx_C.cpp
    GPI_C.cpp
    gaspiEx_C.cpp
    allreduce_C.cpp
 )
 target_link_libraries(PyGPI
        GaspiCxx
        GPI2::GPI2
        )
target_include_directories(PyGPI PUBLIC ${GASPI_CXX_INCLUDE_DIR})
target_link_directories(PyGPI PUBLIC ${GASPI_CXX_LINK_DIR})

target_include_directories (PyGPI
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
	       ${CMAKE_CURRENT_SOURCE_DIR} )


install(TARGETS PyGPI EXPORT
	ARCHIVE  DESTINATION ${CMAKE_INSTALL_DIR}/lib
	LIBRARY  DESTINATION ${CMAKE_INSTALL_DIR}/lib)

if(BUILD_TFOPS)
    add_subdirectory (tf_ops)
endif()