#Create the ComprEx library.
# The extension is already found. Any number of sources could be listed here.
add_library ( ComprEx
    comprexchange.cxx
 )

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
message ("crr source dir is ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (ComprEx
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include 
	       ${CMAKE_CURRENT_SOURCE_DIR} )

## dms install the library, just an example, leave it for later on
## 'make install' to the correct locations (provided by GNUInstallDirs).
#install(TARGETS ACE EXPORT ACEConfig
#	ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#	LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#	RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
# install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## This makes the project importable from the build directory
#export(TARGETS ACE FILE ACEConfig.cmake)
