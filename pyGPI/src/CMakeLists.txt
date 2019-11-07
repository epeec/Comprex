#Create the pyGPI library.
# The extension is already found. Any number of sources could be listed here.
add_library ( pyGPI SHARED
    #ETI.cpp
    ComprEx_C.cpp
    GPI_C.cpp
    GaspiEx_C.cpp
 )
 target_link_libraries(pyGPI
        ${GASPI_CXX_ROOT}/src/libGaspiCxx.a
        GPI2)

add_library ( GPI_C MODULE
       GPI_C.cpp
)
target_link_libraries(GPI_C
       ${GASPI_CXX_ROOT}/src/libGaspiCxx.a
       GPI2)

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
message ("crr source dir is ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (pyGPI
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
	       ${CMAKE_CURRENT_SOURCE_DIR} )

## dms install the library, just an example, leave it for later on
## 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS pyGPI EXPORT
	ARCHIVE  DESTINATION ${CMAKE_INSTALL_DIR}/lib
	LIBRARY  DESTINATION ${CMAKE_INSTALL_DIR}/lib)
install(DIRECTORY ../include DESTINATION ${CMAKE_INSTALL_DIR})

