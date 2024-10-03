# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(MatrixMultiplication)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the Intel C++ Compiler (icx)
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icx)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-tree-vectorize -march=native")

# Find the PAPI library
find_library(PAPI_LIB NAMES papi HINTS /usr/local/lib /usr/lib)

# Find the PAPI include directory
find_path(PAPI_INCLUDE_DIR NAMES papi.h HINTS /usr/local/include /usr/include)

# Check if PAPI was found
if(PAPI_LIB AND PAPI_INCLUDE_DIR)
    message(STATUS "PAPI found: ${PAPI_LIB}, include dir: ${PAPI_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "PAPI not found!")
endif()

# Add executables
add_executable(matrix_multiplication self_attention.cpp)
add_executable(matrix_lars hd_attention.cpp)

# Include PAPI headers in the projects
target_include_directories(matrix_multiplication PRIVATE ${PAPI_INCLUDE_DIR})
target_include_directories(matrix_lars PRIVATE ${PAPI_INCLUDE_DIR})

# Link PAPI and the C++ standard library to the executables
target_link_libraries(matrix_multiplication PRIVATE ${PAPI_LIB} stdc++)
target_link_libraries(matrix_lars PRIVATE ${PAPI_LIB} stdc++)

# Optionally, output some information for debugging
message(STATUS "PAPI include directory: ${PAPI_INCLUDE_DIR}")
message(STATUS "PAPI library: ${PAPI_LIB}")

