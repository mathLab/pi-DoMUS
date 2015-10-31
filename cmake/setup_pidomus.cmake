## ---------------------------------------------------------------------
##
## This file is copyrighted by the pidomus authors and by the deal.II
## authors (see below for the original copyright in the deal.II library.)
##
## The structure of the cmake files are the same of those of the
## deal.II library and they have been modified in order to make them
## compatible with the pidomus library.
##
## This file is part of the pi-DoMUS library.
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
# Set up pidomus specific definitions
#
# Definitions marked with *) can be overriden by defining them to cache
# prior to the call of this file. This is done with the help of the
# SET_IF_EMPTY macro.
#
# General information about pidomus:
#
#     PIDOMUS_PACKAGE_NAME            *)
#     PIDOMUS_PACKAGE_VERSION         *)
#     PIDOMUS_PACKAGE_VENDOR          *)
#     PIDOMUS_PACKAGE_DESCRIPTION     *)
#     PIDOMUS_VERSION_MAJOR
#     PIDOMUS_VERSION_MINOR
#     PIDOMUS_VERSION_SUBMINOR
#     PIDOMUS_VERSION
#
# Information about paths, install locations and names:
#
#     PIDOMUS_PROJECT_CONFIG_NAME     *)
#     PIDOMUS_BASE_NAME               *)
#     PIDOMUS_DEBUG_SUFFIX            *)
#     PIDOMUS_RELEASE_SUFFIX          *)
#
#     PIDOMUS_EXECUTABLE_RELDIR       *)
#     PIDOMUS_INCLUDE_RELDIR          *)
#     PIDOMUS_LIBRARY_RELDIR          *)
#     PIDOMUS_PROJECT_CONFIG_RELDIR   *)
#     PIDOMUS_SHARE_RELDIR            *)
#     PIDOMUS_DOCREADME_RELDIR        *)
#     PIDOMUS_DOCHTML_RELDIR          *)
#     PIDOMUS_EXAMPLES_RELDIR         *)
#
#     PIDOMUS_BUILD_TYPES
#
# *)  Can be overwritten by the command line via -D<...>
#

########################################################################
#                                                                      #
#                  General information about pidomus:                #
#                                                                      #
########################################################################

SET_IF_EMPTY(PIDOMUS_PACKAGE_NAME "pidomus")

SET_IF_EMPTY(PIDOMUS_PACKAGE_VENDOR
  "The pidomus Authors <https://github.com/mathLab/pi-DoMUS/>"
  )
SET_IF_EMPTY(PIDOMUS_PACKAGE_DESCRIPTION
  "A Toolkit library for deal.II <http://www.dealii.org/>"
  )

FILE(STRINGS "${CMAKE_SOURCE_DIR}/VERSION" _version LIMIT_COUNT 1)
SET_IF_EMPTY(PIDOMUS_PACKAGE_VERSION "${_version}")

#
# We expect a version number of the form "X.Y.Z", where X and Y are always
# numbers and Z is either a third number (for a release version) or a short
# string.
#
STRING(REGEX REPLACE "^([0-9]+)\\..*" "\\1"
  PIDOMUS_VERSION_MAJOR "${PIDOMUS_PACKAGE_VERSION}"
  )
STRING(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1"
  PIDOMUS_VERSION_MINOR "${PIDOMUS_PACKAGE_VERSION}"
  )

#
# If Z is not a number, replace it with "0", otherwise extract version
# number:
#
IF(PIDOMUS_PACKAGE_VERSION MATCHES "^[0-9]+\\.[0-9]+.*\\.[0-9]+.*")
  STRING(REGEX REPLACE "^[0-9]+\\.[0-9]+.*\\.([0-9]+).*" "\\1"
    PIDOMUS_VERSION_SUBMINOR "${PIDOMUS_PACKAGE_VERSION}"
    )
ELSE()
  SET(PIDOMUS_VERSION_SUBMINOR "0")
ENDIF()
SET(PIDOMUS_VERSION ${PIDOMUS_VERSION_MAJOR}.${PIDOMUS_VERSION_MINOR}.${PIDOMUS_VERSION_SUBMINOR})


########################################################################
#                                                                      #
#         Information about paths, install locations and names:        #
#                                                                      #
########################################################################

SET(PIDOMUS_PROJECT_CONFIG_NAME "${PIDOMUS_PACKAGE_NAME}")

STRING(REPLACE "." "_" _base_name "${PIDOMUS_PACKAGE_NAME}")
SET_IF_EMPTY(PIDOMUS_BASE_NAME "${_base_name}")
SET_IF_EMPTY(PIDOMUS_DEBUG_SUFFIX ".g")
SET_IF_EMPTY(PIDOMUS_RELEASE_SUFFIX "")

#
# Try to obey the FSHS as close as possible ...
#
SET_IF_EMPTY(PIDOMUS_EXECUTABLE_RELDIR "bin")
SET_IF_EMPTY(PIDOMUS_INCLUDE_RELDIR "include")
SET_IF_EMPTY(PIDOMUS_LIBRARY_RELDIR "lib${LIB_SUFFIX}")
SET_IF_EMPTY(PIDOMUS_PROJECT_CONFIG_RELDIR "${PIDOMUS_LIBRARY_RELDIR}/cmake/${PIDOMUS_PROJECT_CONFIG_NAME}")
SET_IF_EMPTY(PIDOMUS_SHARE_RELDIR "share/${PIDOMUS_PACKAGE_NAME}")
#
# ... but install the documentation into prominent places:
#
SET_IF_EMPTY(PIDOMUS_DOCREADME_RELDIR "./")
SET_IF_EMPTY(PIDOMUS_DOCHTML_RELDIR "doc")
# SET_IF_EMPTY(PIDOMUS_EXAMPLES_RELDIR "examples")

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
  LIST(APPEND PIDOMUS_BUILD_TYPES "DEBUG")
ENDIF()

IF(CMAKE_BUILD_TYPE MATCHES "Release")
  LIST(APPEND PIDOMUS_BUILD_TYPES "RELEASE")
ENDIF()


SET(PIDOMUS_LIST_SUFFIXES
  DEFINITIONS DEFINITIONS_RELEASE DEFINITIONS_DEBUG
  USER_DEFINITIONS USER_DEFINITIONS_RELEASE USER_DEFINITIONS_DEBUG
  INCLUDE_DIRS USER_INCLUDE_DIRS BUNDLED_INCLUDE_DIRS
  LIBRARIES LIBRARIES_RELEASE LIBRARIES_DEBUG
  )

SET(PIDOMUS_STRING_SUFFIXES
  CXX_FLAGS CXX_FLAGS_RELEASE CXX_FLAGS_DEBUG
  LINKER_FLAGS LINKER_FLAGS_RELEASE LINKER_FLAGS_DEBUG
  )
