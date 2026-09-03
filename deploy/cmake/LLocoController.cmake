include_guard(GLOBAL)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation prefix")
find_path(
  ONNXRUNTIME_INCLUDE_DIR
  onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include"
)
find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found; set ONNXRUNTIME_ROOT")
endif()

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(
  onnxruntime::onnxruntime
  PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

function(lloco_add_controller target source needs_cnpy)
  file(GLOB_RECURSE controller_sources CONFIGURE_DEPENDS "${source}/src/*.cpp")
  add_library(${target}_lib STATIC ${controller_sources})
  target_compile_features(${target}_lib PUBLIC cxx_std_17)
  target_include_directories(
    ${target}_lib
    PUBLIC
      "${source}/include"
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../include"
  )
  target_link_libraries(
    ${target}_lib
    PUBLIC
      Boost::program_options
      Eigen3::Eigen
      fmt::fmt
      onnxruntime::onnxruntime
      spdlog::spdlog
      unitree_sdk2
      yaml-cpp
      ddsc
      ddscxx
      pthread
      rt
  )

  if(needs_cnpy)
    find_package(ZLIB REQUIRED)
    add_library(
      ${target}_cnpy STATIC
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../thirdparty/cnpy/cnpy.cpp"
    )
    target_include_directories(
      ${target}_cnpy PUBLIC
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../thirdparty/cnpy"
    )
    target_link_libraries(${target}_cnpy PUBLIC ZLIB::ZLIB)
    target_link_libraries(${target}_lib PUBLIC ${target}_cnpy)
  endif()

  add_executable(${target} "${source}/main.cpp")
  target_link_libraries(${target} PRIVATE ${target}_lib)
endfunction()
