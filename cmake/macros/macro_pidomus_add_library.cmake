## ---------------------------------------------------------------------
##
## This file is copyrighted by the pidomus authors and by the deal.II
## authors (see below for the original copyright in the deal.II library.)
##
## The structure of the cmake files are the same of those of the
## deal.II library and they have been modified in order to make them
## compatible with the pidomus library.
##
## This file is part of the pidomus library.
##
## The pidomus library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the pidomus distribution.
##
## ---------------------------------------------------------------------

## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2015 by the deal.II authors
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
# A small wrapper around ADD_LIBRARY that will define a target for each
# build type specified in PIDOMUS_BUILD_TYPES
#
# It is assumed that the desired compilation configuration is set via
#   PIDOMUS_LINKER_FLAGS_${build}
#   PIDOMUS_CXX_FLAGS_${build}
#   PIDOMUS_DEFINITIONS_${build}
#
# as well as the global (for all build types)
#   PIDOMUS_LINKER_FLAGS
#   PIDOMUS_CXX_FLAGS
#   PIDOMUS_DEFINITIONS
#

MACRO(PIDOMUS_ADD_LIBRARY _library)

  FOREACH(_build ${PIDOMUS_BUILD_TYPES})
    STRING(TOLOWER ${_build} _build_lowercase)

    ADD_LIBRARY(${_library}.${_build_lowercase}
      ${ARGN}
      )

    SET_TARGET_PROPERTIES(${_library}.${_build_lowercase} PROPERTIES
      LINK_FLAGS "${PIDOMUS_LINKER_FLAGS} ${PIDOMUS_LINKER_FLAGS_${_build}}"
      COMPILE_DEFINITIONS "${PIDOMUS_DEFINITIONS};${PIDOMUS_DEFINITIONS_${_build}}"
      COMPILE_FLAGS "${PIDOMUS_CXX_FLAGS} ${PIDOMUS_CXX_FLAGS_${_build}}"
      LINKER_LANGUAGE "CXX"
      )

    SET_PROPERTY(GLOBAL APPEND PROPERTY PIDOMUS_OBJECTS_${_build}
      "$<TARGET_OBJECTS:${_library}.${_build_lowercase}>"
      )

    #STRING(TOUPPER "${_build_type}" _BUILD_TYPE)
    #DEAL_II_SETUP_TARGET(${_library}.${_build_lowercase} ${_BUILD_TYPE})
    DEAL_II_SETUP_TARGET(${_library}.${_build_lowercase} ${_build})
    D2K_SETUP_TARGET(${_library}.${_build_lowercase} ${_build})
  ENDFOREACH()

ENDMACRO()
