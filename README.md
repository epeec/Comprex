# ComprEx

1. INTRODUCTION
===============

ComprEx is an application library, GaspiCxx-based, allowing to exchange large vectors (which are "sparse", in a certain sense) after compressing them first.
This library - supposedly - meets the needs of Machine Learning Applications.

2. INSTALLATION
===============

Requirements:
-------------
 (i) cmake-version > 3.6 ( presently build using 3.9.3) 
 (ii) c++ 14 (presently with gcc-5.2.0 on seislab)

Building comprex
----------------
2.1. clone the git-repository into <comrex_root>

2.2. edit appropriatelly <comrex_root>/CMakeFiles.txt to set there the variables
 -> GASPI_CXX_ROOT
 -> if GPI-2 is not to be loaded as a module, redefine PKG_CONFIG_PATH by 
    adding the path to the file GPI2.pc (the package-config file for GPI-2)
 -> eventually, comment the line set (CMAKE_SHARED_LINKER_FLAGS ..)
    ( it has been added due to the relative old g++ system-libraries )

2.3. in <comprex_root> create a subdirectory "build" to compile comprex
  $ cd  <comrex_root>
  $ mkdir build
  $ cd build
  $ cmake .. -DCMAKE_INSTALL_DIR=<target_installation_dir>
  $ make install

After building and installing comprex, 

-> the library "libComprEx.a" is installed in <target_installation_dir>/lib
-> the header "comprex.hxx" is in <target_installation_dir>/include
-> the executable "example0" is in <comrex_root>/build/examples

