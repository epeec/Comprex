#add executable called "example0" that is built from the source file
# "example.cxx". The extensions are automatically found.
add_executable (example0 example0.cpp)

target_include_directories (ComprEx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link the executable to the Hello library. Since the Hello library has
# public include directories we will use those link directories when building
# helloDemo

#if (GPI2-eth)
#  target_link_libraries (HelmholtzSolver LINK_PUBLIC 
#                           FemHelmholtz 
#			   metis 
#			   gaspils 
#			   numa 
#			   ${GPI2_LIBRARIES} 
#			   pthread 
#			   rt)
#
#else (GPI2-eth)
  target_link_libraries (example0 LINK_PUBLIC 
                           ComprEx 
			   ${GPI2_LIBRARIES} 
			   pthread 
                           ibverbs
			   rt)
#endif (GPI2-eth)
