#Create the pyGPI library.
# The extension is already found. Any number of sources could be listed here.
add_library ( tfGPI SHARED
    ops.cc
 )
 target_link_libraries(tfGPI
        GaspiCxx
        GPI2
        pthread
        ibverbs
		rt)

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
#message ("crr source dir is ${CMAKE_CURRENT_SOURCE_DIR}")
#target_include_directories (tfGPI
#    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
#	       ${CMAKE_CURRENT_SOURCE_DIR} )

## dms install the library, just an example, leave it for later on
## 'make install' to the correct locations (provided by GNUInstallDirs).
#install(TARGETS tfGPI EXPORT
#	ARCHIVE  DESTINATION ${CMAKE_INSTALL_DIR}/lib
#	LIBRARY  DESTINATION ${CMAKE_INSTALL_DIR}/lib)
#install(DIRECTORY ../include DESTINATION ${CMAKE_INSTALL_DIR})

