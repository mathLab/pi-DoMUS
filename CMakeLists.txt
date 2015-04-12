##
#  CMake script for the stokes tutorial program:
##
# cmake_minimum_required(VERSION 2.8)

# project (gradient)
# file(GLOB cc_files source/*.c*)
# include_directories(include)
# add_executable(gradient ${cc_files})

# Set the name of the project and target:
SET(TARGET "stoke")
SET(_main source/main.cc)
# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
 # FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
 file(GLOB _files source/*cc)
 # FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
 include_directories(include)
 # SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
# SET(TARGET_SRC
#   ${TARGET_SRC} 
#   ${TARGET}.cc
#   )
SET(TARGET_SRC
    ${_files}
  )
  
  IF(NOT "${_main}" STREQUAL "") 
      LIST(REMOVE_ITEM _files
        ${CMAKE_CURRENT_SOURCE_DIR}/${_main}
      )
  ENDIF()
# Define the output that should be cleaned:
SET(CLEAN_UP_FILES *.vtu *.pvtu *.visit)

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.3 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fullfilled?
#
IF( NOT DEAL_II_WITH_MPI OR
    NOT DEAL_II_WITH_P4EST OR
    NOT DEAL_II_WITH_TRILINOS )
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_TRILINOS = ON
One or all of these are OFF in your installation but are required for this tutorial step."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  TARGET_LINK_LIBRARIES(${TARGET} $ENV{DEAL_II_SAK}/lib/libdealii-sak.g.dylib)
  # TARGET_LINK_LIBRARIES(${TARGET} $ENV{DEAL_II_SAK}/lib/libdealii-sak.g.so)
ELSE()
  TARGET_LINK_LIBRARIES(${TARGET} $ENV{DEAL_II_SAK}/lib/libdealii-sak.dylib)
  # TARGET_LINK_LIBRARIES(${TARGET} $ENV{DEAL_II_SAK}/lib/libdealii-sak.so)
ENDIF()

SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES "dylib")
# SET(CMAKE_FIND_LIBRARY_SUFFIXES "so")

INCLUDE_DIRECTORIES($ENV{DEAL_II_SAK}/include)
LINK_DIRECTORIES($ENV{DEAL_II_SAK}/lib)

# MacOS
FIND_LIBRARY(libdealii-sak.dylib libdealii-sak.dylib PATHS $ENV{DEAL_II_SAK}/lib )
FIND_LIBRARY(libdealii-sak.g.dylib libdealii-sak.g.dylib PATHS $ENV{DEAL_II_SAK}/lib )
# # Linux
# FIND_LIBRARY(libdealii-sak.so libdealii-sak.so PATHS $ENV{DEAL_II_SAK}/lib )
# FIND_LIBRARY(libdealii-sak.g.so libdealii-sak.g.so PATHS $ENV{DEAL_II_SAK}/lib )

IF(DEAL_II_BUILD_TYPE MATCHES "Release")
  ADD_LIBRARY(${TARGET}-lib SHARED ${_files})
  IF(NOT "${_main}" STREQUAL "") 
  	# ADD_EXECUTABLE(${TARGET} ${_main})
  	TARGET_LINK_LIBRARIES(${TARGET} ${TARGET}-lib)
  	DEAL_II_SETUP_TARGET(${TARGET} RELEASE)
  ENDIF()
  SET(TEST_LIBRARIES_RELEASE ${TARGET}-lib)
  DEAL_II_SETUP_TARGET(${TARGET}-lib RELEASE)
ENDIF()

IF(DEAL_II_BUILD_TYPE MATCHES "Debug")
  ADD_LIBRARY(${TARGET}-lib.g SHARED ${_files})
  IF(NOT "${_main}" STREQUAL "") 
  	ADD_EXECUTABLE(${TARGET}.g ${_main})
  	# TARGET_LINK_LIBRARIES(${TARGET}.g ${TARGET}-lib.g)
  	# DEAL_II_SETUP_TARGET(${TARGET}.g DEBUG)
  ENDIF()
  SET(TEST_LIBRARIES_DEBUG ${TARGET}-lib.g)
  DEAL_II_SETUP_TARGET(${TARGET}-lib.g DEBUG)
ENDIF()

# ADD_SUBDIRECTORY(tests)
# ENABLE_TESTINGm()
