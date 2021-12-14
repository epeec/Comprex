cmake_minimum_required (VERSION 3.6)
project (PyGPI)

###########################
# USER Pathes
###########################
# Adjust here!
set (GASPI_CXX_ROOT "")
option(BUILD_TFOPS "Build the Tensorflow operations for Comprex" OFF)
###########################
option(LINK_IB "Defines whether to link against Infiniband drivers [default: enabled]" OFF)
option(PROFILE "Switch for adding profiling options flags" OFF)
option(DEBUG "Switch to add debugging flags" OFF)
###########################

set(CMAKE_VERBOSE_MAKEFILE "Set makefile verbosity" FALSE)

set(CMAKE_INSTALL_RPATH $ORIGIN)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_INSTALL_DIR "${CMAKE_SOURCE_DIR}")
set(COMPREX_ROOT "${CMAKE_SOURCE_DIR}")

include(GNUInstallDirs)
include_directories(${COMPREX_ROOT}/src)
include_directories(${COMPREX_ROOT}/include)

execute_process(COMMAND which python OUTPUT_VARIABLE PYTHON)
string(STRIP ${PYTHON} PYTHON)

set (CMAKE_BUILD_TYPE Release)

set (COMPREX_SRC "${CMAKE_SOURCE_DIR}/src")
set (COMPREX_BUILD "${CMAKE_SOURCE_DIR}/build")

# Enable Profiling with gprof
if(PROFILE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g -fno-inline")
    set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg -g")
endif()

# enable debugging
if(DEBUG)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if(BUILD_TFOPS)
    set (CONDA_ENVIRONMENT $ENV{CONDA_DEFAULT_ENV})
    message ("Using Conda environment: ${CONDA_ENVIRONMENT}")
    find_package(Tensorflow REQUIRED)
endif()

find_package(GPI2 REQUIRED)

# include GASPI_CXX
set(GASPI_CXX_INCLUDE_DIR ${GASPI_CXX_ROOT}/include)
set(GASPI_CXX_LINK_DIR ${GASPI_CXX_ROOT}/build/src)
include_directories ( ${GASPI_CXX_INCLUDE_DIR} )
link_directories ( ${GASPI_CXX_LINK_DIR} )
message ("GASPI_CXX include: ${GASPI_CXX_INCLUDE_DIR}")
message ("GASPI_CXX lib: ${GASPI_CXX_LINK_DIR}")

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (test)

# create dummy nodefile
file(WRITE ${COMPREX_BUILD}/nodelist "localhost\nlocalhost\n")

