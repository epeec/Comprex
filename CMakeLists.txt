cmake_minimum_required (VERSION 3.6)
project (ComprEx)

include(GNUInstallDirs)

# switch for enabling unit tests. GTEST_ROOT directory
# needs to be set.  default: ON
set (Test OFF)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DETI_DOUBLE -DETI_INT -Wall -Wextra -pedantic")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
set (GTEST_ROOT "path-to-gtest")

set (GASPI_CXX_ROOT "${PROJECT_SOURCE_DIR}")

# add GPI2
# if GPI-2 has NOT been loaded as a module, then set appropriately 
# the environment variable $PKG_CONFIG_PATH in the row below
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{GASPI_HOME}/lib64/pkgconfig")

FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(GPI2 REQUIRED GPI2>=1.3.0)
include_directories ( ${GPI2_INCLUDE_DIRS} )
link_directories ( ${GPI2_LIBRARY_DIRS} )

include_directories (
        ${GPI2_INCLUDE_DIRS} 
        ${GASPI_CXX_ROOT}/include
        )

link_directories (
        ${GPI2_LIBRARY_DIRS} 
        ${GASPI_CXX_ROOT}/lib
        )
	
add_subdirectory (src)
add_subdirectory (examples)

if (Test)
  add_subdirectory (tests)
endif (Test)  
