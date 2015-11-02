#!/bin/sh

PRG=$PWD/programs
CASA=$PWD

if [ ! -d programs ] 
then
  echo "create folder `$PRG`"
  mkdir $PRG
else
  # touch all files to avoid cache to be deleted
  find $PRG -exec touch {} \;
fi

# installing numdiff
if [ ! -d $PRG/numdiff ]
then
  echo "installing numdiff"
  mkdir $PRG/numdiff-tmp
  cd $PRG/numdiff-tmp
  wget http://mirror.lihnidos.org/GNU/savannah//numdiff/numdiff-5.8.1.tar.gz
  tar xzf numdiff-5.8.1.tar.gz
  rm numdiff-5.8.1.tar.gz
  cd numdiff-5.8.1
  DST_INST=$PRG/numdiff
  ./configure --prefix=$DST_INST > /dev/null
  make -j4 install > /dev/null
  cd $CASA
  rm -rf $PRG/numdiff-tmp
fi

# trilinos
if [ ! -d $PRG/trilinos ]
then
  echo "installing trilinos"
  DST_INST=$PRG/trilinos
  export PATH=$PRG/cmake/bin:$PATH
  cd $PRG
  git clone https://github.com/trilinos/trilinos.git trilinos-tmp
  cd trilinos-tmp
  mkdir build
  cd build
  cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE:STRING=RELEASE \
    -D TPL_ENABLE_Boost:BOOL=OFF \
    -D TPL_ENABLE_BoostLib:BOOL=OFF \
    -D TPL_ENABLE_MPI:BOOL=OFF \
    -D TPL_ENABLE_Netcdf:BOOL=OFF \
    -D CMAKE_INSTALL_PREFIX:PATH=$DST_INST \
    -D Trilinos_ENABLE_OpenMP:BOOL=OFF \
    -D BUILD_SHARED_LIBS:BOOL=ON \
    -D Trilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="" \
    -D CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE \
    -D Trilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF \
    -D Trilinos_ENABLE_TESTS:BOOL=OFF \
    -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
    -D Trilinos_ENABLE_Epetra:BOOL=ON \
    -D Trilinos_ENABLE_NOX:BOOL=ON \
    -D Trilinos_ENABLE_EpetraExt:BOOL=ON \
    -D Trilinos_ENABLE_Tpetra:BOOL=ON \
    -D Trilinos_ENABLE_Kokkos:BOOL=ON \
    -D Trilinos_ENABLE_Sacado:BOOL=ON \
    -D Trilinos_ENABLE_Amesos:BOOL=ON \
    -D Trilinos_ENABLE_AztecOO:BOOL=ON \
    -D Trilinos_ENABLE_Ifpack:BOOL=ON \
    -D Trilinos_ENABLE_Rythmos:BOOL=ON \
    -D Trilinos_ENABLE_Piro:BOOL=ON \
    -D Trilinos_ENABLE_MOOCHO:BOOL=ON \
    -D Trilinos_ENABLE_ML:BOOL=ON \
    -D Trilinos_ENABLE_MueLu:BOOL=ON \
    -D Trilinos_ENABLE_Komplex:BOOL=ON \
    -D Trilinos_ENABLE_Thyra:BOOL=ON \
    -D Trilinos_ENABLE_TrilinosCouplings:BOOL=ON \
    -D Trilinos_ENABLE_Fortran:BOOL=OFF \
    -D CMAKE_CXX_COMPILER:PATH=/usr/bin/clang++-3.6 \
    -D CMAKE_CXX_FLAGS:STRING=-w \
    -D CMAKE_C_COMPILER:PATH=/usr/bin/clang-3.6 \
    -D CMAKE_C_FLAGS:STRING=-w \
    .. > $CASA/trilinos_cmake.log 2>&1

  ninja -j4 
  ninja -j4 install > /dev/null
  cd $PRG
  rm -rf trilinos-tmp
  cd $CASA
fi

# dealii
if [ ! -d $PRG/dealii ]
then
  echo "installing dealii"
  DST_INST=$PRG/dealii 
  cd $PRG
  git clone https://github.com/dealii/dealii.git dealii-tmp
  cd dealii-tmp
  mkdir build
  cd build
  cmake \
    -G Ninja \
    -D CMAKE_INSTALL_PREFIX:PATH=$DST_INST \
    -D CMAKE_CXX_FLAGS:STRING=-w \
    -D DEAL_II_WITH_MPI:BOOL=OFF \
    -D DEAL_II_WITH_THREADS:BOOL=OFF \
    .. #> $CASA/dealii_cmake.log 2>&1
  ninja -j3 
  ninja -j4 install > /dev/null
  cd $PRG
  rm -rf dealii-tmp
  cd $CASA
fi

# deal2lkit
if [ ! -d $PRG/deal2lkit ]
then
  echo "installing deal2lkit"
  DST_INST=$PRG/deal2lkit 
  cd $PRG
  git clone https://github.com/mathlab/deal2lkit.git deal2lkit-tmp
  cd deal2lkit-tmp
  mkdir build
  cd build
  cmake \
    -G Ninja \
    -D CMAKE_INSTALL_PREFIX:PATH=$DST_INST \
    .. 
  ninja -j4 install
  cd $PRG
  rm -rf deal2lkit-tmp
  cd $CASA
fi

