cmake_minimum_required(VERSION 3.8)
project(pedcom_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceRegion.msg"
  "msg/EmotionalState.msg"
  "msg/DistressLevel.msg"
  "msg/MultisensoryCommand.msg"
  "msg/BehaviourStatus.msg"
  "srv/SetBehaviour.srv"
  "srv/StartSession.srv"
  "action/ComfortSession.action"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
