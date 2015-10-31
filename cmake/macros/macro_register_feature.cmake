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
## Copyright (C) 2013 - 2015 by the deal.II authors
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
# This macro is used for the feature configuration in pidomus. It adds
# individual FEATURE_* configuration variables to the corresponding
# PIDOMUS_* variables
#
# Usage:
#     REGISTER_FEATURE(feature)
#
# This macro will add
#
#   <FEATURE>_LIBRARIES (respecting general, optimized, debug keyword)
#
# and all other suffixes defined in PIDOMUS_LIST_SUFFIXES and
# PIDOMUS_STRING_SUFFIXES to the corresponding PIDOMUS_* variables
#

MACRO(REGISTER_FEATURE _feature)

  IF(DEFINED ${_feature}_LIBRARIES)
    #
    # Add ${_feature}_LIBRARIES to
    #   PIDOMUS_LIBRARIES
    #   PIDOMUS_LIBRARIES_DEBUG
    #   PIDOMUS_LIBRARIES_RELEASE
    # depending on the "optmized", "debug" or "general" keyword
    #
    SET(_toggle "general")
    FOREACH(_tmp ${${_feature}_LIBRARIES})
      IF( "${_tmp}" STREQUAL "debug" OR
          "${_tmp}" STREQUAL "optimized" OR
          "${_tmp}" STREQUAL "general" )
        SET(_toggle "${_tmp}")
      ELSE()
        IF("${_toggle}" STREQUAL "general")
          LIST(APPEND PIDOMUS_LIBRARIES ${_tmp})
        ELSEIF("${_toggle}" STREQUAL "debug")
          LIST(APPEND PIDOMUS_LIBRARIES_DEBUG ${_tmp})
        ELSEIF("${_toggle}" STREQUAL "optimized")
          LIST(APPEND PIDOMUS_LIBRARIES_RELEASE ${_tmp})
        ENDIF()
      ENDIF()
    ENDFOREACH()
  ENDIF()

  FOREACH(_var ${PIDOMUS_LIST_SUFFIXES})
    IF(NOT "${_var}" STREQUAL "LIBRARIES" AND DEFINED ${_feature}_${_var})
      LIST(APPEND PIDOMUS_${_var} ${${_feature}_${_var}})
    ENDIF()
  ENDFOREACH()

  FOREACH(_var ${PIDOMUS_STRING_SUFFIXES})
    IF(DEFINED ${_feature}_${_var})
      ADD_FLAGS(PIDOMUS_${_var} "${${_feature}_${_var}}")
    ENDIF()
  ENDFOREACH()

ENDMACRO()
