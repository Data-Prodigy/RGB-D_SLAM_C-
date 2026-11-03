cmake_minimum_required(VERSION 3.24)
project(rgbd_slam VERSION 1.0.0 DESCRIPTION "My humble trial implementation of RGBD Slam from scratch with the KinectV1" LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)   # before any target

#Forcing out-of-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "In-source builds are not allowed, please create a build directory and run CMAKE from there.")
endif()

#Setting my default build type
get_property(_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "NO build type specified, defaulting to release...")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

#setting my C++23 standard requirement and disabling compiler-specific extensions
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 2. Forcing the CUDA compiler (nvcc) to use C++17 to match my .cu files
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# 3. Explicitly target compute_75 (Turing) and generate PTX/binary for sm_75
set(CMAKE_CUDA_ARCHITECTURES "75")

#making sure intellisense and autocomplete work well
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SLEEKSLAM_USE_SANITIZERS "ASan/UBSan (GCC/Clang only)" OFF)
option(SLEEKSLAM_BUILD_TESTS "Build unit tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Dependencies)

#helps my compiled binary ctch memory bugs at runtime/UBs e.g signed integer overflows, null pointer derefences etc...
if(SLEEKSLAM_USE_SANITIZERS AND NOT MSVC)
	add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>") #last option helps sanitizer point exactly to the line that caused a bug
	add_link_options(-fsanitize=address,undefined)
endif()

#CUDA CODE
add_library(sleekslam_cuda STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/sleekslam/odometry/residuals_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/sleekslam/mapping/tsdf_gpu.cu
)
set_target_properties(sleekslam_cuda PROPERTIES VS_GLOBAL_VCToolsVersion 14.38.33130)
target_precompile_headers(sleekslam_cuda PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_pch.h"
)
target_include_directories(sleekslam_cuda PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
    "C:/dev/glew-install/include"
    "C:/dev/eigen-5.0.0"
    "C:/Program Files/fbow/include"
    ${Pangolin_INCLUDE_DIRS} # Added to resolve pangolin/pangolin.h
)
target_link_libraries(sleekslam_cuda PUBLIC
    Eigen3::Eigen
    pango_core
    pango_display
    pango_geometry
    pango_glgeometry
    pango_image
    pango_opengl
    pango_packetstream
    pango_plot
    pango_scene
    pango_tools
    pango_vars
    pango_video
    pango_windowing
)
add_library(sleekslam INTERFACE)
#now #include "include/camera.hpp", "mapping/fusion.hpp" resolve from this root
target_include_directories(sleekslam INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${FBOW_INCLUDE_DIR}")
target_compile_features(sleekslam INTERFACE cxx_std_23)
target_link_libraries(sleekslam INTERFACE
    sleekslam_cuda

    Eigen3::Eigen
    Sophus::Sophus
    ${OpenCV_LIBS}

    GLEW::glew

    pango_core
    pango_display
    pango_geometry
    pango_glgeometry
    pango_image
    pango_opengl
    pango_packetstream
    pango_plot
    pango_scene
    pango_tools
    pango_vars
    pango_video
    pango_windowing

    ${FBOW_LIBRARY}

    g2o::core
    g2o::stuff
    g2o::types_slam3d
    g2o::solver_eigen
    g2o::solver_dense
    g2o::solver_csparse
)
add_executable(main src/main.cpp)
set_target_properties(main PROPERTIES VS_GLOBAL_VCToolsVersion 14.44.35207)
target_precompile_headers(main PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h)

#private here çause nothing ever links against my .exe
target_link_libraries(main PRIVATE sleekslam project_warnings)
# Copy all runtime DLL dependencies beside the executable (Windows).
if(WIN32)
    find_file(FBOW_DLL NAMES fbow001.dll
        HINTS "C:/Program Files/fbow" PATH_SUFFIXES bin lib REQUIRED)
        
    find_file(OPENCV_WORLD_DLL NAMES opencv_world4140.dll
        HINTS "C:/dev/opencv_final_build/x64/vc17" PATH_SUFFIXES bin REQUIRED)

    # Copy manual DLLs and vcpkg runtime DLL dependencies in one clean sequence
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FBOW_DLL}"
            "${OPENCV_WORLD_DLL}"
            "$<TARGET_FILE_DIR:main>"
        COMMAND ${CMAKE_COMMAND} -E copy -t 
            "$<TARGET_FILE_DIR:main>" 
            $<TARGET_RUNTIME_DLLS:main>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying fbow, OpenCV, and runtime DLL dependencies..."
    )
endif()
if(SLEEKSLAM_BUILD_TESTS)
  enable_testing()                          # must be in the root
  add_subdirectory(tests)
endif()