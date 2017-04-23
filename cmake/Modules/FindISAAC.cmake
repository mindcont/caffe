SET(ISAAC_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/local/include
  /opt/isaac/include
  $ENV{ISAAC_HOME}
  $ENV{ISAAC_HOME}/include
)

SET(ISAAC_LIB_SEARCH_PATHS
        /lib
        /lib64
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/isaac/lib
        $ENV{ISAAC_HOME}
        $ENV{ISAAC_HOME}/lib
 )

FIND_LIBRARY(ISAAC_LIBRARY NAMES isaac PATHS ${ISAAC_LIB_SEARCH_PATHS})

SET(ISAAC_FOUND ON)

#    Check libraries
IF(NOT ISAAC_LIBRARY)
    SET(ISAAC_FOUND OFF)
    MESSAGE(STATUS "Could not find ISAAC lib. Turning ISAAC_FOUND off")
ENDIF()

IF (ISAAC_FOUND)
  IF (NOT ISAAC_FIND_QUIETLY)
    MESSAGE(STATUS "Found ISAAC libraries: ${ISAAC_LIBRARY}")
    MESSAGE(STATUS "Found ISAAC include: ${ISAAC_INCLUDE_DIR}")
  ENDIF (NOT ISAAC_FIND_QUIETLY)
ELSE (ISAAC_FOUND)
  IF (ISAAC_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find ISAAC")
  ENDIF (ISAAC_FIND_REQUIRED)
ENDIF (ISAAC_FOUND)

MARK_AS_ADVANCED(
    ISAAC_INCLUDE_DIR
    ISAAC_LIBRARY
)

