# Replay recorded insertion operators, scores and gate decisions.
add_executable(test_lges_operators test_lges_operators.cpp)
target_link_libraries(test_lges_operators PRIVATE xges_core)
add_test(NAME lges_operators
         COMMAND test_lges_operators ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

# Boundary conventions that sampled data cannot reach.
add_executable(test_lges_semantics test_lges_semantics.cpp)
target_link_libraries(test_lges_semantics PRIVATE xges_core)
add_test(NAME lges_semantics COMMAND test_lges_semantics)

# Real search trajectories, graph invariants, score validation and cache ownership.
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE xges_core)
add_test(NAME core_regressions COMMAND test_core ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)
set_tests_properties(core_regressions PROPERTIES TIMEOUT 120)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME cli_regressions COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py $<TARGET_FILE:xges>)
set_tests_properties(cli_regressions PROPERTIES TIMEOUT 180 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_test(NAME experiment_regressions COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_experiments.py)
set_tests_properties(experiment_regressions PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    foreach(target test_lges_operators test_lges_semantics test_core)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
endif()
