cmake_minimum_required(VERSION 3.16)
project(cloud_reg)

find_package(PCL 1.11 REQUIRED COMPONENTS common io visualization filters segmentation surface registration)
find_package(teaserpp REQUIRED)
find_package(OpenGR REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pc_0 pc_0.cpp ply_loader.cpp poly_gen.cpp)
target_link_libraries(pc_0 ${PCL_LIBRARIES})

add_executable(pc_1 pc_1.cpp CloudReg.cpp poly_gen.cpp point_sampler.cpp)
target_link_libraries(pc_1 ${PCL_LIBRARIES})

add_executable(pc_2 pc_2/try_icp.cpp poly_gen.cpp point_sampler.cpp)
target_link_libraries(pc_2 ${PCL_LIBRARIES})

add_executable(pc_2_hulls pc_2/try_icp_hulls.cpp)
target_link_libraries(pc_2_hulls ${PCL_LIBRARIES})

add_executable(pc_2_teaser pc_2/try_teaser.cpp)
target_link_libraries(pc_2_teaser ${PCL_LIBRARIES} teaserpp::teaser_registration teaserpp::teaser_io teaserpp::teaser_features)

add_executable(pc_2_seg pc_2/try_seg.cpp poly_gen.cpp point_sampler.cpp)
target_link_libraries(pc_2_seg ${PCL_LIBRARIES})

add_executable(pc_2_teaser_ransac pc_2/try_teaser_ransac.cpp)
target_link_libraries(pc_2_teaser_ransac ${PCL_LIBRARIES} teaserpp::teaser_registration teaserpp::teaser_io teaserpp::teaser_features)

add_executable(pc_2_teaser_ransac_i pc_2/try_teaser_ransac_i.cpp)
target_link_libraries(pc_2_teaser_ransac_i ${PCL_LIBRARIES} teaserpp::teaser_registration teaserpp::teaser_io teaserpp::teaser_features)

add_executable(sample pc_2/sample.cpp point_sampler.cpp)
target_link_libraries(sample ${PCL_LIBRARIES})

add_executable(pc_2_opengr pc_2/try_opengr.cpp)
target_link_libraries(pc_2_opengr ${PCL_LIBRARIES} gr::algo)

add_executable(pc_2_opengr_mt pc_2/try_opengr_multithread.cpp)
target_link_libraries(pc_2_opengr_mt ${PCL_LIBRARIES} gr::algo)

add_executable(compare pc_2/compare.cpp)
target_link_libraries(compare ${PCL_LIBRARIES} gr::algo teaserpp::teaser_registration teaserpp::teaser_io teaserpp::teaser_features)

add_executable(gencorr pc_2/gencorr.cpp)
target_link_libraries(gencorr ${PCL_LIBRARIES})

add_executable(pc_2_rt pc_2/test_robustness.cpp)
target_link_libraries(pc_2_rt ${PCL_LIBRARIES} gr::algo)

add_executable(pc_3 pc_3.cpp)
target_link_libraries(pc_3 ${PCL_LIBRARIES})

add_executable(pc_3_1 pc_3_1.cpp)
target_link_libraries(pc_3_1 ${PCL_LIBRARIES})