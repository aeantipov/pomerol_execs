cmake_minimum_required (VERSION 2.8.12)
project (pomerol_execs CXX)


# Disable build in source
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory.")
endif()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# find pomerol
find_package(pomerol REQUIRED)
message(STATUS "pomerol includes : ${pomerol_INCLUDE_DIRS}")
message(STATUS "pomerol libs : ${pomerol_LIBRARIES}")
include_directories(${pomerol_INCLUDE_DIRS})
link_libraries(${pomerol_LIBRARIES})

#find alpscore
find_package(ALPSCore REQUIRED COMPONENTS params hdf5)
message(STATUS "Found ALPSCore")
include_directories(${ALPSCORE_INCLUDE_DIR})
link_libraries(${ALPSCore_LIBRARIES})

# find gftools
find_package(GFTools REQUIRED)
message(STATUS "GFTools includes : ${GFTOOLS_INCLUDE_DIR}")
include_directories(${GFTOOLS_INCLUDE_DIR})

set (execs anderson)
foreach (exec ${execs})
    add_executable(${exec} ${exec}.cpp)
    target_link_libraries(${exec} ${pomerol_LIBRARIES})
endforeach (exec)
