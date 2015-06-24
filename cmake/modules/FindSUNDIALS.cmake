
find_path (SUNDIALS_DIR include/sundials/sundials_config.h HINTS ENV SUNDIALS_DIR PATHS $ENV{HOME}/sundials DOC "Sundials Directory")

IF(EXISTS ${SUNDIALS_DIR}/include/sundials/sundials_config.h)
  SET(SUNDIALS_FOUND YES)
  SET(SUNDIALS_INCLUDES ${SUNDIALS_DIR})
  find_path (SUNDIALS_INCLUDE_DIR sundials_config.h HINTS "${SUNDIALS_DIR}" PATH_SUFFIXES include/sundials NO_DEFAULT_PATH)
  list(APPEND SUNDIALS_INCLUDES ${SUNDIALS_INCLUDE_DIR})
  FILE(GLOB _SUNDIALS_LIBRARIES "${SUNDIALS_DIR}/lib" "${SUNDIALS_DIR}/lib/libsundials*.a")
 FOREACH(_lib ${_SUNDIALS_LIBRARIES})
     FIND_LIBRARY(SUNDIALS_${_lib} ${_lib} HINTS ${SUNDIALS_DIR} PATH_SUFFIXES lib) 
     IF(NOT ${SUNDIALS_${_lib}} STREQUAL "${SUNDIALS_${_lib}}-NOTFOUND")
        MESSAGE("SUNDIALS_${_lib}:  ${SUNDIALS_${_lib}}")
	LIST(APPEND SUNDIALS_LIBRARIES ${SUNDIALS_${_lib}})
     ENDIF()
 ENDFOREACH()
ELSE(EXISTS ${SUNDIALS_DIR}/include/sundials/sundials_config.h)
  SET(SUNDIALS_FOUND NO)
  message(FATAL_ERROR "Cannot find SUNDIALS!")
ENDIF(EXISTS ${SUNDIALS_DIR}/include/sundials/sundials_config.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SUNDIALS DEFAULT_MSG SUNDIALS_LIBRARIES SUNDIALS_INCLUDES)

