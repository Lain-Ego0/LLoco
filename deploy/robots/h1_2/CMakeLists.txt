cmake_minimum_required(VERSION 3.18)
project(h1_2_controller LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/LLocoController.cmake")
lloco_add_controller(h1_2_ctrl "${CMAKE_CURRENT_SOURCE_DIR}" FALSE)
