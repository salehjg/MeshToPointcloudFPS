#include <iostream>
#include "settings.hpp"
#include "cudahelper.hpp"
#include "pcl.hpp"
#include "hdf5.hpp"
#include "argparse.h"

using namespace std;
using namespace argparse;

void initCuda(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    cout<<"Selected CUDA Device: "<< dev<<", "<< deviceProp.name<<endl;
    CHECK(cudaSetDevice(dev));
    CHECK(cudaDeviceReset());
    CHECK(cudaDeviceSynchronize());
}

int main(int argc, const char** argv){
    ArgumentParser parser("fps_cuda");

    parser.add_argument()
        .names({"-i", "--inputmesh"})
        .description("The path for the input mesh *.obj file.")
        .required(true); 

    parser.add_argument()
        .names({"-o", "--outputhdf5"})
        .description("The path for the output hdf5 *.h5 file with sampled point cloud.")
        .required(true);

    parser.add_argument()
        .names({"-n", "--npoints"})
        .description("The target number of points per mesh input file (input.obj).")
        .required(true);  

    parser.add_argument()
        .names({"-p", "--outputpcd"})
        .description("The path for the optional output pcd *.pcd file with raw point cloud.")
        .required(false);       

    parser.enable_help();
    auto err = parser.parse(argc, argv);
    if(err){
        std::cerr << err << std::endl;
        parser.print_help();
        return -1;
    }

    if(parser.exists("help")){
        parser.print_help();
        return 0;
    }


    initCuda();
    
}

/*
    const unsigned long lenImage = CONFIG_IMAGE_HEIGHT * CONFIG_IMAGE_WIDTH * 3;
    unsigned char *d_src1;
    unsigned char *h_dst1;
    unsigned char *d_dst1;

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    cout<<"Selected CUDA Device: "<< dev<<", "<< deviceProp.name<<endl;
    CHECK(cudaSetDevice(dev));
    CHECK(cudaDeviceReset());
    CHECK(cudaDeviceSynchronize());

    h_dst1 = (unsigned char*)malloc(sizeof(unsigned char) * lenImage);
    CHECK(cudaMalloc((void**)&d_mask, sizeof(float) * lenMask));

    CHECK(cudaMemcpy(d_src1, frame.data, sizeof(unsigned char) * lenImage, cudaMemcpyHostToDevice));
    LaunchKernel_SpatialFilter(
            d_src1,
            d_mask,
            d_dst1);
    CHECK(cudaMemcpy(h_dst1, d_dst1, sizeof(unsigned char) * lenImage, cudaMemcpyDeviceToHost));
    cudaFree(d_dst1);
    free(h_dst1);
*/