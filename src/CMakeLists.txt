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
