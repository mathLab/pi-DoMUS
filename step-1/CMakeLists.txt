cmake_minimum_required(VERSION 3.0)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

set(_exe "step-1")

FIND_PACKAGE(deal.II 8.3 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )



DEAL_II_INITIALIZE_CACHED_VARIABLES()


FIND_PACKAGE(deal2lkit 1.0 REQUIRED
  HINTS ${D2K_DIR} $ENV{D2K_DIR} $ENV{DEAL2LKIT_DIR}
  )
D2K_INITIALIZE_CACHED_VARIABLES()


find_package(pidomus  REQUIRED
  HINTS ${PIDOMUS_DIR})

project(step-1)

# set(CMAKE_)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${_exe} poisson_01.cc)
    # SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

    D2K_SETUP_TARGET(${_exe})
    target_link_libraries(${_exe} pidomus-lib)

