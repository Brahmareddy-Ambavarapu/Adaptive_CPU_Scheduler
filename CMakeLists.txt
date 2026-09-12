cmake_minimum_required(VERSION 3.10)
project(adaptive-cpu-scheduler)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(cpp/include)

add_library(cpu_scheduler_lib
    cpp/src/process.cpp
    cpp/src/simulator.cpp
    cpp/src/metrics.cpp
    cpp/src/schedulers/fcfs.cpp
    cpp/src/schedulers/sjf.cpp
    cpp/src/schedulers/rr.cpp
    cpp/src/schedulers/priority.cpp
    cpp/src/workload_generator.cpp
    cpp/src/dataset_generator.cpp
    cpp/src/workload_feature_extractor.cpp
    cpp/src/rule_based_adaptive_scheduler.cpp
    cpp/src/ml_adaptive_scheduler.cpp
    cpp/src/oracle_scheduler.cpp
    cpp/src/experiment_runner.cpp
)

# Main Executable
add_executable(scheduler cpp/src/main.cpp)
target_link_libraries(scheduler PRIVATE cpu_scheduler_lib)

# Enable C++ testing
enable_testing()
find_package(GTest REQUIRED)

# Test Targets
add_executable(test_process tests/test_process.cpp)
target_link_libraries(test_process PRIVATE cpu_scheduler_lib GTest::gtest_main)
add_test(NAME test_process COMMAND test_process)

add_executable(test_schedulers tests/test_schedulers.cpp)
target_link_libraries(test_schedulers PRIVATE cpu_scheduler_lib GTest::gtest_main)
add_test(NAME test_schedulers COMMAND test_schedulers)

add_executable(test_workload_generator tests/test_workload_generator.cpp)
target_link_libraries(test_workload_generator PRIVATE cpu_scheduler_lib GTest::gtest_main)
add_test(NAME test_workload_generator COMMAND test_workload_generator)

add_executable(test_dataset_generator tests/test_dataset_generator.cpp)
target_link_libraries(test_dataset_generator PRIVATE cpu_scheduler_lib GTest::gtest_main)
add_test(NAME test_dataset_generator COMMAND test_dataset_generator)

add_executable(test_rule_based_adaptive_scheduler tests/test_rule_based_adaptive_scheduler.cpp)
target_link_libraries(test_rule_based_adaptive_scheduler PRIVATE cpu_scheduler_lib)
add_test(NAME test_rule_based_adaptive_scheduler COMMAND test_rule_based_adaptive_scheduler)

add_executable(test_ml_adaptive_scheduler tests/test_ml_adaptive_scheduler.cpp)
target_link_libraries(test_ml_adaptive_scheduler PRIVATE cpu_scheduler_lib)
add_test(NAME test_ml_adaptive_scheduler COMMAND test_ml_adaptive_scheduler)

add_executable(test_oracle_scheduler tests/test_oracle_scheduler.cpp)
target_link_libraries(test_oracle_scheduler PRIVATE cpu_scheduler_lib)
add_test(NAME test_oracle_scheduler COMMAND test_oracle_scheduler)

add_executable(test_experiment_runner tests/test_experiment_runner.cpp)
target_link_libraries(test_experiment_runner PRIVATE cpu_scheduler_lib)
add_test(NAME test_experiment_runner COMMAND test_experiment_runner)
