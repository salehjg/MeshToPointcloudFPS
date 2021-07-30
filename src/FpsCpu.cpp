#include <iostream>
#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "argparse.h"
#include "pcl.hpp"
#include "hdf5.hpp"

using namespace argparse;
using namespace std;

void FpsCpuSingleThread(const int n, const int m, const float *input, float *output){
    float *temp = new float[n];
    int old=0;
    output[0]=input[0];
    output[1]=input[1];
    output[2]=input[2];
    for (int j=0;j<n;j++){
      temp[j]=1e38;
    }
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      float x1=input[old*3+0], y1=input[old*3+1], z1=input[old*3+2];
      for (int k=0;k<n;k++){
        float x2=input[k*3+0], y2=input[k*3+1], z2=input[k*3+2];
        float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
        float d2=min(d,temp[k]);
        if (d2!=temp[k])
          temp[k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }
      old=besti;
      output[j*3+0]=input[old*3+0];
      output[j*3+1]=input[old*3+1];
      output[j*3+2]=input[old*3+2];
    }
    delete[] temp;
    return;
}

int main(int argc, const char** argv){
    ArgumentParser parser("FpsCpu", "FpsCpu");

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
        .names({"-r", "--rawpcd"})
        .description("The path for the optional output pcd *.pcd file with RAW point cloud.")
        .required(false);  

    parser.add_argument()
        .names({"-p", "--outputpcd"})
        .description("The path for the optional output pcd *.pcd file with sampled point cloud.")
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


    const string pathObj(parser.get<string>("i"));
    const string pathPcd(parser.exists("r")?parser.get<string>("r"):"");

    CMesh2Pcl *m_oMesh2Pcl = new CMesh2Pcl(pathObj, pathPcd);

    m_oMesh2Pcl->Convert();
    
    const int nRaw = m_oMesh2Pcl->GetTensorShape()[0];
    const int nDest = parser.get<int>("n");
    float *buffRaw = new float[nRaw*3];
    float *buffSampled = new float[nDest*3];
    m_oMesh2Pcl->CloneToBuffRowMajor(buffRaw);

    FpsCpuSingleThread(nRaw,nDest,buffRaw,buffSampled);

    if(parser.exists("p")){
        m_oMesh2Pcl->DumpToXyzPcd((const float*)buffSampled, nDest, parser.get<string>("p"));
    }

    exportHdf5(parser.get<string>("o").c_str(),buffSampled,nDest,3);
    
    delete[] buffRaw;
    delete[] buffSampled;
    delete m_oMesh2Pcl;

    return 0;
}

