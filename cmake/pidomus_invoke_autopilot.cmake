## This macro is a wrapper for the DEAL_II_INVOKE_AUTOPILOT() CMake
## macro written by the deal.II authors. We report it's documentation

## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2016 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------

#
# This file implements the DEAL_II_INVOKE_AUTOPILOT macro, which is
# part of the deal.II library.
#
# Usage:
#       DEAL_II_INVOKE_AUTOPILOT()
#
# where it is assumed that the following variables are defined:
#
#       TARGET         -  a string used for the project and target name
#       TARGET_SRC     -  a list of source file to compile for target
#                         ${TARGET}
#       TARGET_RUN     -  (optional) the command line that should be
#                         invoked by "make run", will be set to default
#                         values if undefined. If no run target should be
#                         created, set it to an empty string.
#       CLEAN_UP_FILES -  (optional) a list of files (globs) that will be
#                         removed with "make runclean" and "make
#                         distclean", will be set to default values if
#                         empty

#### specific for the pidomus library

#       PARAMTER_FILE - the parameter file which has to be sourced. If
#                       the executable is built and run in a different
#                       folder (e.g. build) a symlink is done
#
macro(PIDOMUS_INVOKE_AUTOPILOT)

DEAL_II_INVOKE_AUTOPILOT()

SET(_d2_build_types "Release" "Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

FOREACH(_build_type ${_d2_build_types})
  SET(_p "${${_build_type}_postfix}")
  
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      PIDOMUS_BUILD_TYPE MATCHES "${_build_type}")

    # set(exe "${_exe}${_p}")
    # add_executable(${TARGET} ${TARGET_SRC})
    # SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

    string(TOUPPER "${_build_type}" _BUILD_TYPE)
    

    D2K_SETUP_TARGET(${TARGET} ${_BUILD_TYPE})
    set(_lib "pidomus-lib${_p}")
    target_link_libraries(${TARGET} ${_lib})
  endif()
endforeach()

# if we build out of source (e.g. build) we create a link to the
# parameter file
if(NOT EXISTS "${CMAKE_BINARY_DIR}/${PARAMETER_FILE}")
  execute_process(COMMAND
    ln -s ${CMAKE_SOURCE_DIR}/${PARAMETER_FILE}
    ${CMAKE_BINARY_DIR}/${PARAMETER_FILE})
endif()

endmacro()
