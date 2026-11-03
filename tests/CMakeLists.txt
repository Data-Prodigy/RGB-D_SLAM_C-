include(FetchContent)
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.14.0 GIT_SHALLOW TRUE)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(slam_pose_tests test_pose_graph.cpp)
set_target_properties(slam_pose_tests PROPERTIES VS_GLOBAL_VCToolsVersion 14.44.35207)

target_link_libraries(slam_pose_tests PRIVATE GTest::gtest_main sleekslam)

include(GoogleTest)

gtest_discover_tests(slam_pose_tests)