cmake_minimum_required(VERSION 3.12)
project(fps_cuda)
set(CMAKE_CXX_STANDARD 11)

set(REPO_DIR_CMAKE  ${PROJECT_SOURCE_DIR})
set(BIN_DIR_CMAKE   ${CMAKE_BINARY_DIR})

SET(CUDA_INCLUDE_DIRS /opt/cuda/include)
SET(CUDA_LIBRARIES cudart cublas)
SET(CUDA_LIBRARY_DIRS /opt/cuda/lib64)

find_package(CUDA REQUIRED)

SET(CUDA_HOST_COMPILER /usr/bin/gcc-10)
set(CUDA_PROPAGATE_HOST_FLAGS False)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

find_package(PCL REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)

# Find all cuda kernels
#(https://stackoverflow.com/questions/29308787/cmake-file-for-cpp-cu-files)
file( GLOB  KERNEL_SOURCES  ${REPO_DIR_CMAKE}/kernels/cuda/*.cu)
message("Found these kernels: ")
message("${KERNEL_SOURCES}")

include_directories(
        ${PCL_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
        ${REPO_DIR_CMAKE}/inc
        ${REPO_DIR_CMAKE}/argparse
        )

cuda_add_executable(fps_cuda
        ${KERNEL_SOURCES}
        ${REPO_DIR_CMAKE}/src/main.cpp
        )

target_link_libraries(fps_cuda ${KERNEL_OBJs} ${PCL_LIBRARIES} ${HDF5_LIBRARIES})