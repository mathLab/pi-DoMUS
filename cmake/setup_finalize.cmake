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



########################################################################
#                                                                      #
#                      Finalize the configuration:                     #
#                                                                      #
########################################################################
#
#
# Hide some cmake specific cached variables. This is annoying...
#
MARK_AS_ADVANCED(file_cmd)

#
# Append the saved initial (cached) variables ${flags}_SAVED at the end of
# ${flags}, see setup_cached_compiler_flags.cmake and the main
# CMakeLists.txt for details.
#
#  FOREACH(_flags ${PIDOMUS_USED_FLAGS})
#    # Strip leading and trailing whitespace:
#    STRING(STRIP "${${_flags}} ${${_flags}_SAVED}" ${_flags})
#  ENDFOREACH()
#
#  #
#  # Sanity check: The variables defined in PIDOMUS_REMOVED_FLAGS must not be
#  # used during the configuration stage:
#  #
#  FOREACH(_flag ${PIDOMUS_REMOVED_FLAGS})
#    IF(NOT "${${_flag}}" STREQUAL "")
#      MESSAGE(FATAL_ERROR
#        "\nInternal configuration error: The variable ${_flag} was set to a "
#        "non empty value during the configuration! (The corresponding "
#        "PIDOMUS_* variable should have been used.)\n"
#        "${_flag}=\"${${_flag}}\"\n"
#        )
#    ENDIF()
#  ENDFOREACH()

#
# Save base configuration into variables BASE_* for later use in
# setup_write_config.cmake:
#
FOREACH(_suffix ${PIDOMUS_STRING_SUFFIXES} ${PIDOMUS_LIST_SUFFIXES})
  SET(BASE_${_suffix} ${PIDOMUS_${_suffix}})
ENDFOREACH()

#
# Register features:
#
FOREACH(_feature ${PIDOMUS_FEATURES})
  FILTER_SYSTEM_LIBRARIES(${_feature}) # TODO, remove here
  REGISTER_FEATURE(${_feature})
ENDFOREACH()

#
# Deduplicate entries one more time :-]
#
FOREACH(_suffix ${PIDOMUS_LIST_SUFFIXES})
  IF(_suffix MATCHES "INCLUDE_DIRS$")
    REMOVE_DUPLICATES(PIDOMUS_${_suffix})
  ELSE()
    REMOVE_DUPLICATES(PIDOMUS_${_suffix} REVERSE)
  ENDIF()
ENDFOREACH()


# Clean up pidomusTargets.cmake in the build directory:

FILE(REMOVE
  ${CMAKE_BINARY_DIR}/${PIDOMUS_PROJECT_CONFIG_RELDIR}/${PIDOMUS_PROJECT_CONFIG_NAME}Targets.cmake
  )
