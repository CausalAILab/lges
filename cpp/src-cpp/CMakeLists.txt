set(BINARY xges)

# The algorithm lives in a library so that both the CLI and the tests link it.
add_library(
        xges_core STATIC
        PDAG.cpp
        PDAG.h
        ScorerInterface.h
        Operators.h
        Operators.cpp
        set_ops.h
        set_ops.cpp
        EdgeQueueSet.cpp
        EdgeQueueSet.h
        XGES.cpp
        XGES.h
        LGES.cpp
        LGES.h
        BICScorer.cpp
        BICScorer.h
        DataIO.h
        DataIO.cpp
        utils.h
)
target_include_directories(xges_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(xges_core PUBLIC cxx_std_23)
target_compile_definitions(xges_core PUBLIC EIGEN_USE_BLAS)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(xges_core PRIVATE -Wall -Wextra -Wpedantic -Wimplicit-fallthrough)
endif()

target_link_libraries(xges_core PUBLIC spdlog::spdlog)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost REQUIRED)
target_link_libraries(xges_core PUBLIC Boost::boost)

# Sources include "Eigen/Dense", so the include dir must be the eigen3 root.
find_package(Eigen3 REQUIRED)
target_link_libraries(xges_core PUBLIC Eigen3::Eigen)

find_package(BLAS REQUIRED)
target_include_directories(xges_core PUBLIC ${BLAS_INCLUDE_DIRS})
target_link_libraries(xges_core PUBLIC ${BLAS_LIBRARIES})

add_executable(${BINARY} main.cpp)
target_link_libraries(${BINARY} PRIVATE xges_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${BINARY} PRIVATE -Wall -Wextra -Wpedantic)
endif()
