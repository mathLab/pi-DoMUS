#!/bin/sh

PRG=$PWD/programs
CASA=$PWD

# deal2lkit
echo "-------------------------------------->installing deal2lkit"
cd $CASA
DST_INST=$CASA/deal2lkit
git clone https://github.com/mathlab/deal2lkit.git deal2lkit-src
cd deal2lkit-src
mkdir build
cd build
cmake \
    -G Ninja \
    -D CMAKE_INSTALL_PREFIX:PATH=$DST_INST-$BUILD_TYPE \
    -D CMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
    -D D2K_COMPONENT_EXAMPLES:BOOL=OFF \
    -D D2K_COMPONENT_DOCUMENTATION:BOOL=OFF \
    .. 
ninja -j4 install
cd $CASA

