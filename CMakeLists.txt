cmake_minimum_required(VERSION 3.16)
project(cloud_reg)

find_package(PCL 1.3 REQUIRED COMPONENTS common io visualization filters segmentation surface)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pc_0 pc_0.cpp ply_loader.cpp poly_gen.cpp)
target_link_libraries(pc_0 ${PCL_LIBRARIES})

add_executable(pc_1 pc_1.cpp InputHandler.cpp InputManager.cpp CloudReg.cpp poly_gen.cpp)
target_link_libraries(pc_1 ${PCL_LIBRARIES})
