cmake_minimum_required (VERSION 3.6)
project (PyGPI)

###############################################################
### USER SECTION
### MODIFY THIS SECTION ACCORDINGLY

set (BASE_DIR "/home/loroch/PROJECTS")

# the googletest directory
set (GTEST_ROOT "${BASE_DIR}/googletest")

# the GaspiCxx directory
set (GASPI_CXX_ROOT "${BASE_DIR}/GaspiCxx")

# switch for enabling unit tests. GTEST_ROOT directory
# needs to be set.  default: ON
set (Test OFF)

# enable profiling of C++ code
set (PROFILE OFF)

# OPTIONAL: add GPI2 explicitely
# if the GPI-2 does not appear in your PKG_CONFIG_PATH environment variable, then set
# the environment variable $PKG_CONFIG_PATH in the row below
# set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:<your_path_to_GPI-2>/lib64/pkgconfig")

# OPTIONAL: change installation directory of comprex
# the default installation directory is /build
set (CMAKE_INSTALL_DIR "${CMAKE_SOURCE_DIR}/build")

###############################################################

set (COMPREX_ROOT "${CMAKE_SOURCE_DIR}")

include(GNUInstallDirs)

execute_process(COMMAND which python OUTPUT_VARIABLE PYTHON)
string(STRIP ${PYTHON} PYTHON)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DETI_DOUBLE -DETI_INT -Wall -pedantic -fpic")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")

set (COMPREX_SRC "${CMAKE_SOURCE_DIR}/src")
set (COMPREX_BUILD "${CMAKE_SOURCE_DIR}/build")

#set (CMAKE_SHARED_LINKER_FLAGS "-Wl,-rpath=/home/work/soft/gcc/5.2.0/lib")

# Enable Profiling with gprof
if(PROFILE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g")
    set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg -g")
endif()

FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(GPI2 REQUIRED GPI2>=1.3.0)
include_directories ( ${GPI2_INCLUDE_DIRS} )
link_directories ( ${GPI2_LIBRARY_DIRS} )
message ("GPI2 include: ${GPI2_INCLUDE_DIRS}")
message ("GPI2 lib: ${GPI2_LIBRARY_DIRS}")

include_directories (
        ${GPI2_INCLUDE_DIRS}
        ${GASPI_CXX_ROOT}/include
        )

link_directories (
        ${GPI2_LIBRARY_DIRS}
        ${GASPI_CXX_ROOT}/src
        )

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (test)

if (Test)
  add_subdirectory (tests)
endif (Test)



