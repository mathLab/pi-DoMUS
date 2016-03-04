##
# CMake script for large project, with one executable
# and a library of objects depending on deal2lkit.
#
# The structure of the directory is assumed to be: 
# ./source: containing implementations and no files with "int main()"
# ./include: containing all class declarations
# ./tests: containing pairs of files: test.cc/test.output to test
# ./main: containing files with "int main()", which are compiled to
# one executable each
#
# If you set the environemnt variable DEAL2LKIT_DIR or D2K_DIR, 
# everything will work out of the box
##

SET(TARGET pidomus)

# Declare all source files the target consists of:
file(GLOB _files source/*cc)
file(GLOB _mains main/*cc)


INCLUDE_DIRECTORIES(include)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.3 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()


PROJECT(${TARGET})
FIND_PACKAGE(deal2lkit 1.0 REQUIRED
  HINTS ${D2K_DIR} $ENV{D2K_DIR} $ENV{DEAL2LKIT_DIR}
  )
D2K_INITIALIZE_CACHED_VARIABLES()
OPTION(ENABLE_APP_TESTING "Enable tests in this application." ON)
OPTION(ENABLE_DOCUMENTATION "Enable Doxygen documentation." ON)

# We add one library and one target for each type of deal.II library
# we found. If you compiled deal.II with both Release and Debug
# mode, this will generate both Release and Debug programs for you.
# The debug library and program are postfixed with ".g"
SET(_d2_build_types "Release;Debug")
SET(Release_postfix "")
SET(Debug_postfix ".g")

FOREACH(_build_type ${_d2_build_types})
  # Postfix to use everywhere
  SET(_p "${${_build_type}_postfix}")
  # Only build this type, if deal.II was compiled with it.
  IF(CMAKE_BUILD_TYPE MATCHES "${_build_type}" AND
      DEAL_II_BUILD_TYPE MATCHES "${_build_type}"  AND
      D2K_BUILD_TYPE MATCHES "${_build_type}")

    MESSAGE("-- Found ${_build_type} version of deal.II.")
    MESSAGE("-- Found ${_build_type} version of deal2lkit.")

    SET(_lib "${TARGET}-lib${_p}")
    MESSAGE("-- Configuring library ${_lib}")

    STRING(TOUPPER "${_build_type}" _BUILD_TYPE)

    ADD_LIBRARY(${_lib} SHARED ${_files})
    FOREACH(_main ${_mains})
	MESSAGE("-- Processing ${_main}")
	GET_FILENAME_COMPONENT(_app_name ${_main} NAME_WE)
    	SET(_exe "${_app_name}${_p}")
    	MESSAGE("-- Configuring executable ${_exe}")
    	ADD_EXECUTABLE(${_exe} ${_main})
    	TARGET_LINK_LIBRARIES(${_exe} ${_lib})
	D2K_SETUP_TARGET(${_exe} ${_BUILD_TYPE})
    ENDFOREACH()

    SET(TEST_LIBRARIES_${_BUILD_TYPE} ${_lib})

    D2K_SETUP_TARGET(${_lib} ${_BUILD_TYPE})
    SET_TARGET_PROPERTIES(${_lib} PROPERTIES MACOSX_RPATH OFF
	BUILD_WITH_INSTALL_RPATH OFF
	INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib
    )
  ENDIF()
ENDFOREACH()


if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests AND ${ENABLE_APP_TESTING})
  ADD_SUBDIRECTORY(tests)
  ENABLE_TESTING()
endif()

IF(ENABLE_DOCUMENTATION)
  MESSAGE(STATUS "Setting up documentation")
  # Doxygen
  FIND_PACKAGE(Doxygen)
  IF(DOXYGEN_FOUND)
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    ADD_CUSTOM_TARGET(doc
      ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen" VERBATIM
      )
  MESSAGE(STATUS "Setting up documentation - Done")
  ENDIF()
ENDIF()


ADD_CUSTOM_TARGET(indent
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./scripts/indent
  )

