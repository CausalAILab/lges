cmake_minimum_required(VERSION 3.26)
project(xges)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(lib)
add_subdirectory(src-cpp)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
