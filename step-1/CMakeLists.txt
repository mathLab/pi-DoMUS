cmake_minimum_required(VERSION 3.0)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

set(_exe "step-1")

FIND_PACKAGE(deal.II 8.5 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

set(CMAKE_BUILD_TYPE "${DEAL_II_BUILD_TYPE}" CACHE STRING "")

DEAL_II_INITIALIZE_CACHED_VARIABLES()


FIND_PACKAGE(deal2lkit 1.0 REQUIRED
  HINTS ${D2K_DIR} $ENV{D2K_DIR} $ENV{DEAL2LKIT_DIR}
  )
D2K_INITIALIZE_CACHED_VARIABLES()


find_package(pidomus 8.5 REQUIRED
  HINTS ${PIDOMUS_DIR})

project(${_exe})

# set(CMAKE_)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

FOREACH(_build_type ${_d2_build_types})
    SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}"  AND
      D2K_BUILD_TYPE MATCHES "${_build_type}")

    set(exe "${_exe}${_p}")
    add_executable(${exe} poisson_01.cc)
    # SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

    string(TOUPPER "${_build_type}" _BUILD_TYPE)
    

    D2K_SETUP_TARGET(${exe} ${_BUILD_TYPE})
    set(_lib "pidomus-lib${_p}")
    target_link_libraries(${exe} ${_lib})
  endif()
endforeach()

