##
#  CMake script for the step-19 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "test_MC_3_parallel")


# Declare all source files the target consists of:
SET(TARGET_SRC
  #Project.cpp
  #test_Stokes.cpp
  test_Navier_Stokes.cpp
  #test_MC.cpp
  #test_knn.cpp
  RandomGenerator.cpp
  # You can specify additional files here!
  )

# No sensible run target can be defined for this example step
SET(TARGET_RUN "")

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE( deal.II 8.0 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR} )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# find_library( lapack_lib lapacke)


DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
ADD_CUSTOM_TARGET( run COMMAND ${TARGET} COMMENT "Run with ${CMAKE_BUILD_TYPE} configuration" )
DEAL_II_INVOKE_AUTOPILOT()
# TARGET_LINK_LIBRARIES(${TARGET} ${lapack_lib} )


# FLANN for k-NN search
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(FLANN 1.8 REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS})
add_definitions(${FLANN_DEFINITIONS})
target_link_libraries (${TARGET} ${FLANN_LIBRARIES})


# pcl for k-NN search
#find_package(PCL 1.7 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})
