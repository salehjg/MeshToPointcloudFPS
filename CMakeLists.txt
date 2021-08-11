cmake_minimum_required(VERSION 3.12)
project(FpsCpu)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(REPO_DIR_CMAKE  ${PROJECT_SOURCE_DIR})
set(BIN_DIR_CMAKE   ${CMAKE_BINARY_DIR})

find_package(PCL REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
find_package( Eigen3 REQUIRED )
add_subdirectory(vcglib) # C++ Header-only lib

include_directories(
        ${PCL_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        ${REPO_DIR_CMAKE}/inc
        ${REPO_DIR_CMAKE}/argparse
        vcglib
        ${EIGEN3_INCLUDE_DIR}
        )

add_executable(FpsCpu 
        ${REPO_DIR_CMAKE}/src/FpsCpu.cpp
        ${REPO_DIR_CMAKE}/src/CMeshToPcl.cpp
        ${REPO_DIR_CMAKE}/src/hdf5.cpp
        ${REPO_DIR_CMAKE}/src/CPureTriMesh.cpp)

target_link_libraries(FpsCpu ${PCL_LIBRARIES} ${HDF5_LIBRARIES})