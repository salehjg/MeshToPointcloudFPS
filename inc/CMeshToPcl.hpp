#pragma once

#include <pcl/visualization/pcl_visualizer.h>
#include "pcl_vtk_compatibility.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <vtkVersion.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <string>
#include <vector>
#include "CPureTriMesh.h"

using namespace std;


//http://www1.coe.neu.edu/~ctd/ISY240/CodingGuidelines.htm
class CMeshToPcl{
public:
    CMeshToPcl(string pathObjFile, string pathPcdFile);
    bool FileExist(const std::string& name);
    int Convert();
    int MakeInputObjPureTriMesh();
    int GetTensorSizeWords();
    int GetTensorSizeBytes();
    vector<int> GetTensorShape();
    void CloneToBuffRowMajor(float *outBuff);
    void DumpToXyzPcd(const float *buff, const int nPoints, const string path);
protected:
    double UniformDeviate(int seed);
    void RandomPointTriangle(
        float a1, 
        float a2, 
        float a3, 
        float b1, 
        float b2, 
        float b3, 
        float c1, 
        float c2, 
        float c3,
        float r1, 
        float r2, 
        Eigen::Vector3f& p);
    void RandPSurface(
        vtkPolyData * polydata, 
        std::vector<double> * cumulativeAreas, 
        double totalArea, 
        Eigen::Vector3f& p, 
        bool calcNormal, 
        Eigen::Vector3f& n, 
        bool calcColor, 
        Eigen::Vector3f& c);
    void UniformSampling(
        vtkSmartPointer<vtkPolyData> polydata, 
        std::size_t n_samples, 
        bool calc_normal, 
        bool calc_color, 
        pcl::PointCloud<pcl::PointXYZRGBNormal> & cloud_out);

private:
    string m_strPathRawObjFile, m_strPathObjFile, m_strPathPcdFile;
    bool m_bGeneratePcdFile;
    const int m_iDefaultNumSamples = 100000;
    const float m_fDefaultLeafSize = 0.01f;
    pcl::PointCloud<pcl::PointXYZ> m_oCloudXYZ;
    CPureTriMesh m_oPureTriMesh;
};