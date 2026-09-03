cmake_minimum_required(VERSION 3.18)
project(a2_controller LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/LLocoController.cmake")
lloco_add_controller(a2_ctrl "${CMAKE_CURRENT_SOURCE_DIR}" FALSE)
