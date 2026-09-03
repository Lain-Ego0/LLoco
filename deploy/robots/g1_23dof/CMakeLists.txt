cmake_minimum_required(VERSION 3.18)
project(g1_23dof_controller LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/LLocoController.cmake")
lloco_add_controller(g1_23dof_ctrl "${CMAKE_CURRENT_SOURCE_DIR}" TRUE)
