cmake_minimum_required(VERSION 3.16)
project(unitree_mujoco)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(mujoco CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(Threads REQUIRED)

if(TARGET mujoco::mujoco)
  set(MUJOCO_TARGET mujoco::mujoco)
elseif(TARGET mujoco)
  set(MUJOCO_TARGET mujoco)
else()
  message(FATAL_ERROR "MuJoCo package was found but did not export a mujoco target.")
endif()

if(TARGET glfw)
  set(GLFW_TARGET glfw)
elseif(TARGET glfw3::glfw)
  set(GLFW_TARGET glfw3::glfw)
else()
  message(FATAL_ERROR "GLFW package was found but did not export a supported target.")
endif()

set(SIMULATE_INCLUDE_DIRS
  mujoco/simulate
  src/lodepng
)

file(GLOB SIM_SRC
  src/joystick/joystick.cc

  # exclude mujoco/simulate/main.cc
  mujoco/simulate/glfw_*.cc
  mujoco/simulate/platform_*.cc
  mujoco/simulate/simulate.cc
  src/lodepng/lodepng.cpp
)

add_executable(unitree_mujoco ${SIM_SRC} src/main.cc)
target_include_directories(unitree_mujoco PRIVATE ${SIMULATE_INCLUDE_DIRS})
target_link_libraries(unitree_mujoco PRIVATE
  Threads::Threads
  ${MUJOCO_TARGET}
  ${GLFW_TARGET}
  yaml-cpp
  unitree_sdk2
  Boost::program_options
  fmt::fmt
)

add_executable(jstest src/joystick/jstest.cc src/joystick/joystick.cc)
