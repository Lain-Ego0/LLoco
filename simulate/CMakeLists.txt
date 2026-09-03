cmake_minimum_required(VERSION 3.18)
project(unitree_mujoco LANGUAGES CXX)

include(FetchContent)

set(LLOCO_MUJOCO_TAG "3.3.6" CACHE STRING "MuJoCo source tag used by the bridge")
set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  mujoco
  GIT_REPOSITORY https://github.com/google-deepmind/mujoco.git
  GIT_TAG "${LLOCO_MUJOCO_TAG}"
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(mujoco)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(fmt REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)

file(GLOB simulate_sources CONFIGURE_DEPENDS
  "${mujoco_SOURCE_DIR}/simulate/glfw_*.cc"
  "${mujoco_SOURCE_DIR}/simulate/platform_*.cc"
  "${mujoco_SOURCE_DIR}/simulate/simulate.cc"
)

add_executable(
  unitree_mujoco
  ${simulate_sources}
  src/joystick/joystick.cc
  src/lodepng/lodepng.cpp
  src/main.cc
)
target_compile_features(unitree_mujoco PRIVATE cxx_std_17)
target_include_directories(
  unitree_mujoco
  PRIVATE
    "${mujoco_SOURCE_DIR}/simulate"
    src
    src/lodepng
)
target_link_libraries(
  unitree_mujoco
  PRIVATE
    Boost::program_options
    fmt::fmt
    mujoco::mujoco
    glfw
    pthread
    unitree_sdk2
    yaml-cpp
)

add_executable(jstest src/joystick/jstest.cc src/joystick/joystick.cc)
target_compile_features(jstest PRIVATE cxx_std_17)
