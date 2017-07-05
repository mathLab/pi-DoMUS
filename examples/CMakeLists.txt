message(STATUS "Setting up examples")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION examples
  # COMPONENT examples
  FILES_MATCHING
  #
  # Exclude folder structures: doc, doxygen, CMakeFiles,...
  #
  PATTERN "CMakeFiles*" EXCLUDE
  PATTERN "doc*" EXCLUDE
  PATTERN "build*" EXCLUDE
  #
  # Glob Includes:
  #
  PATTERN "*.cc"
  PATTERN "*.h"
  PATTERN "*.prm"
  PATTERN "*.inp"
  PATTERN "*.output"
  #
  # All CMakeLists.txt except the one in examples/
  #
  PATTERN "heat-equation/CMakeLists.txt"
  #
  # Special files:
  #
  # PATTERN "grid-2.2.ucd" # e.g.
  )

MESSAGE(STATUS "Setting up examples - Done")
