cmake_minimum_required (VERSION 2.8.12)
project (pomerol_hubbard2d CXX)

# Disable build in source
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory.")
endif()
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)


SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#find_package(ALPSCore)
#target_link_libraries(hubbard2d ${ALPSCore_LIBRARIES})

find_package(pomerol)
message(STATUS ${pomerol_LIBRARIES})
message(STATUS ${pomerol_INCLUDE_DIRS})

include_directories(${pomerol_INCLUDE_DIRS})
#include(${POMEROL_ROOT}/share/pomerol/pomerol.cmake)

set (execs anderson)
foreach (exec ${execs})
    add_executable(${exec} ${exec}.cpp)
    target_link_libraries(${exec} ${pomerol_LIBRARIES})
endforeach (exec)
