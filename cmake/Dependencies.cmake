set(EIGEN_ROOT "C:/dev/eigen-5.0.0" CACHE PATH "Folder that contains the Eigen/ header directory")

if(NOT EXISTS "${EIGEN_ROOT}/Eigen/Core")
  message(FATAL_ERROR "Eigen headers not found under ${EIGEN_ROOT}")
endif()

add_library(eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS eigen)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui calib3d features2d)
find_package(Eigen3 CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(Pangolin CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)
find_package(fbow CONFIG REQUIRED)

set(FBOW_INCLUDE_DIR "C:/Program Files/fbow/include")
set(FBOW_LIBRARY "C:/Program Files/fbow/lib/fbow001.lib")

if(NOT EXISTS "${FBOW_LIBRARY}")
    message(FATAL_ERROR "FBoW library not found: ${FBOW_LIBRARY}")
endif() 
target_include_directories(eigen SYSTEM INTERFACE "${EIGEN_ROOT}")