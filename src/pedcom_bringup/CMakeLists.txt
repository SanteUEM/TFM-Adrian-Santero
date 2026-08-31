cmake_minimum_required(VERSION 3.8)
project(pedcom_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch urdf worlds config
        DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/test_pipeline_integration.py TIMEOUT 120)
endif()

ament_package()
