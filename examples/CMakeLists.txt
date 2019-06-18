#add executable called "example0" that is built from the source file
# "example.cxx". The extensions are automatically found.
add_executable (example0 example0.cpp)
add_executable (example1 example1.cpp)

message(" crr src dir:  ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories (example0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link the executable to the Hello library. Since the Hello library has
# public include directories we will use those link directories when building
# helloDemo

target_link_libraries (example0 
                           LINK_PUBLIC "-static-libstdc++"
                           ComprEx 
			   GaspiCxx
			   ${GPI2_LIBRARIES} 
			   pthread 
                           ibverbs
			   rt)

target_link_libraries (example1 
                           LINK_PUBLIC "-static-libstdc++"
                           ComprEx 
			   GaspiCxx
			   ${GPI2_LIBRARIES} 
			   pthread 
                           ibverbs
			   rt)
