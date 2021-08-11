#include "CMeshToPcl.hpp"
#include <sys/stat.h>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace std;

/**
 * @brief Class Constructor.
 * @details .
 * 
 * @param pathObjFile The path to the input mesh file(*.obj) to be read by PCL library.
 * @param pathPcdFile The path to the optional output pointcloud file(*.pcd), will be 
 *                    generated if the value is not empty("").
 */
CMeshToPcl::CMeshToPcl(string pathObjFile, string pathPcdFile){
    m_strPathRawObjFile = pathObjFile;
    m_strPathObjFile.append(m_strPathRawObjFile);
    m_strPathObjFile.append(".tri");
    m_strPathPcdFile = pathPcdFile;
    m_bGeneratePcdFile = !(this->m_strPathPcdFile == "");
}

inline bool CMeshToPcl::FileExist(const std::string& name){
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

/**
 * @brief Converts the input file (mesh) into pointcloud.
 * @details Stores the pointcloud in private 'm_oCloudXYZ'.
 * @return Returns 0 if there was not any errors.
 */
int CMeshToPcl::Convert(){
    int stat = MakeInputObjPureTriMesh();
    if(stat!=0) {
        return stat;
    }

    {
        bool statF=false;
        int timeout = 100;
        while(timeout-->0){
            if(FileExist(m_strPathObjFile)){
                statF = true;
                break;
            }
        }
        if(!statF) return 3;
    }

    vtkSmartPointer<vtkPolyData> polydata1 = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkOBJReader> readerQuery = vtkSmartPointer<vtkOBJReader>::New();
    readerQuery->SetFileName(m_strPathObjFile.c_str());
    readerQuery->Update();
    polydata1 = readerQuery->GetOutput();

    //make sure that the polygons are triangles!
    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(polydata1);
    triangleFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> triangleMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
    triangleMapper->Update();
    polydata1 = triangleMapper->GetInput();

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    UniformSampling(polydata1, m_iDefaultNumSamples, false, false, *cloud_1);

    // Voxelgrid
    VoxelGrid<PointXYZRGBNormal> grid_;
    grid_.setInputCloud(cloud_1);
    grid_.setLeafSize(m_fDefaultLeafSize, m_fDefaultLeafSize, m_fDefaultLeafSize);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    grid_.filter(*voxel_cloud);

    
    // Strip uninitialized normals and colors from cloud:
    m_oCloudXYZ.width    = voxel_cloud->width;
    m_oCloudXYZ.height   = voxel_cloud->height;
    m_oCloudXYZ.is_dense = false;
    m_oCloudXYZ.resize (m_oCloudXYZ.width * m_oCloudXYZ.height);

    pcl::copyPointCloud(*voxel_cloud, m_oCloudXYZ);
    if(m_bGeneratePcdFile){
        savePCDFileASCII(m_strPathPcdFile, m_oCloudXYZ);
    }
    return 0;
}

/**
 * @brief Get the total number of elements in the tensor of the pointcloud.
 * @details .
 * @return .
 */
int CMeshToPcl::GetTensorSizeWords(){
    return 3*m_oCloudXYZ.width*m_oCloudXYZ.height;
}

/**
 * @brief Get the total size of the tensor of the pointcloud in bytes.
 * @details .
 * @return .
 */
int CMeshToPcl::GetTensorSizeBytes(){
    return GetTensorSizeWords()*sizeof(float);
}

/**
 * @brief Get the vector of the pointcloud tensor's shape.
 * @details .
 * @return vector<int>
 */
vector<int> CMeshToPcl::GetTensorShape(){
    vector<int> shape;
    shape.push_back(m_oCloudXYZ.width*m_oCloudXYZ.height);
    shape.push_back(3);
    return shape;
}

/**
 * @brief Get the pointcloud tensor's content into a pre-allocated buffer.
 * @details Takes a pre-allocated buffer which is appropriate for the tensor size
 *          and copies the content of the pointcloud tensor into it with the row-major layout.
 * 
 * @param outBuff A pointer to the pre-allocated buffer of appropriate size for the pointcloud tensor.
 */
void CMeshToPcl::CloneToBuffRowMajor(float *outBuff){
    const int len = GetTensorSizeWords();

    int n=0;
    for (const auto& point : m_oCloudXYZ){
        outBuff[n*3+0]=point.x;
        outBuff[n*3+1]=point.y;
        outBuff[n*3+2]=point.z;
        n++;
    }
}

/**
 * @brief .
 * @details Taken from https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp
 * 
 * @return .
 */
inline double CMeshToPcl::UniformDeviate(int seed){
    double ran = seed *(1.0 /(RAND_MAX + 1.0));
    return ran;
}

/**
 * @brief .
 * @details Taken from https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp
 *
 * @return .
 */
inline void CMeshToPcl::RandomPointTriangle(
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
        Eigen::Vector3f& p){

    float r1sqr = std::sqrt(r1);
    float OneMinR1Sqr =(1 - r1sqr);
    float OneMinR2 =(1 - r2);
    a1 *= OneMinR1Sqr;
    a2 *= OneMinR1Sqr;
    a3 *= OneMinR1Sqr;
    b1 *= OneMinR2;
    b2 *= OneMinR2;
    b3 *= OneMinR2;
    c1 = r1sqr *(r2 * c1 + b1) + a1;
    c2 = r1sqr *(r2 * c2 + b2) + a2;
    c3 = r1sqr *(r2 * c3 + b3) + a3;
    p[0] = c1;
    p[1] = c2;
    p[2] = c3;
}

/**
 * @brief .
 * @details Taken from https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp
 *
 * @return .
 */
inline void CMeshToPcl::RandPSurface(
        vtkPolyData * polydata, 
        std::vector<double> * cumulativeAreas, 
        double totalArea, 
        Eigen::Vector3f& p, 
        bool calcNormal, 
        Eigen::Vector3f& n, 
        bool calcColor, 
        Eigen::Vector3f& c){

    float r = static_cast<float>(UniformDeviate(rand()) * totalArea);

    std::vector<double>::iterator low = std::lower_bound(cumulativeAreas->begin(), cumulativeAreas->end(), r);
    vtkIdType el = vtkIdType(low - cumulativeAreas->begin());

    double A[3], B[3], C[3];
    vtkIdType npts = 0;
    vtkCellPtsPtr ptIds = nullptr;

    polydata->GetCellPoints(el, npts, ptIds);
    polydata->GetPoint(ptIds[0], A);
    polydata->GetPoint(ptIds[1], B);
    polydata->GetPoint(ptIds[2], C);
    if(calcNormal){
        // OBJ: Vertices are stored in a counter-clockwise order by default
        Eigen::Vector3f v1 = Eigen::Vector3f(A[0], A[1], A[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
        Eigen::Vector3f v2 = Eigen::Vector3f(B[0], B[1], B[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
        n = v1.cross(v2);
        n.normalize();
    }
    float r1 = static_cast<float>(UniformDeviate(rand()));
    float r2 = static_cast<float>(UniformDeviate(rand()));
    RandomPointTriangle(float(A[0]), float(A[1]), float(A[2]),
                                             float(B[0]), float(B[1]), float(B[2]),
                                             float(C[0]), float(C[1]), float(C[2]), r1, r2, p);

    if(calcColor){
        vtkUnsignedCharArray *const colors = vtkUnsignedCharArray::SafeDownCast(polydata->GetPointData()->GetScalars());
        if(colors && colors->GetNumberOfComponents() == 3){
            double cA[3], cB[3], cC[3];
            colors->GetTuple(ptIds[0], cA);
            colors->GetTuple(ptIds[1], cB);
            colors->GetTuple(ptIds[2], cC);

            RandomPointTriangle(float(cA[0]), float(cA[1]), float(cA[2]),
                                                     float(cB[0]), float(cB[1]), float(cB[2]),
                                                     float(cC[0]), float(cC[1]), float(cC[2]), r1, r2, c);
        }else{
            static bool printed_once = false;
            if(!printed_once)
                PCL_WARN("Mesh has no vertex colors, or vertex colors are not RGB!\n");
            printed_once = true;
        }
    }
}

/**
 * @brief .
 * @details Taken from https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp
 *
 * @return .
 */
void CMeshToPcl::UniformSampling(
        vtkSmartPointer<vtkPolyData> polydata, 
        std::size_t n_samples, 
        bool calc_normal, 
        bool calc_color, 
        pcl::PointCloud<pcl::PointXYZRGBNormal> & cloud_out){

    polydata->BuildCells();
    vtkSmartPointer<vtkCellArray> cells = polydata->GetPolys();

    double p1[3], p2[3], p3[3], totalArea = 0;
    std::vector<double> cumulativeAreas(cells->GetNumberOfCells(), 0);
    vtkIdType npts = 0;
    vtkCellPtsPtr ptIds = nullptr;
    std::size_t cellId = 0;

    for(cells->InitTraversal(); cells->GetNextCell(npts, ptIds); cellId++){
        polydata->GetPoint(ptIds[0], p1);
        polydata->GetPoint(ptIds[1], p2);
        polydata->GetPoint(ptIds[2], p3);
        totalArea += vtkTriangle::TriangleArea(p1, p2, p3);
        cumulativeAreas[cellId] = totalArea;
    }

    cloud_out.resize(n_samples);
    cloud_out.width = static_cast<std::uint32_t>(n_samples);
    cloud_out.height = 1;

    for(std::size_t i = 0; i < n_samples; i++){
        Eigen::Vector3f p;
        Eigen::Vector3f n(0, 0, 0);
        Eigen::Vector3f c(0, 0, 0);
        RandPSurface(polydata, &cumulativeAreas, totalArea, p, calc_normal, n, calc_color, c);
        cloud_out[i].x = p[0];
        cloud_out[i].y = p[1];
        cloud_out[i].z = p[2];
        if(calc_normal){
            cloud_out[i].normal_x = n[0];
            cloud_out[i].normal_y = n[1];
            cloud_out[i].normal_z = n[2];
        }

        if(calc_color){
            cloud_out[i].r = static_cast<std::uint8_t>(c[0]);
            cloud_out[i].g = static_cast<std::uint8_t>(c[1]);
            cloud_out[i].b = static_cast<std::uint8_t>(c[2]);
        }
    }
}

void CMeshToPcl::DumpToXyzPcd(const float *buff, const int nPoints, const string path){
    pcl::PointCloud<pcl::PointXYZ> cloud;
    // Fill in the cloud data
    cloud.width    = nPoints;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.resize(cloud.width * cloud.height);

    int i=0;
    for (auto& point: cloud){
        point.x = buff[i*3+0];
        point.y = buff[i*3+1];
        point.z = buff[i*3+2];
        i++;
    }
    pcl::io::savePCDFileASCII (path, cloud);
}

int CMeshToPcl::MakeInputObjPureTriMesh() {
    return m_oPureTriMesh.MakeObjFilePureTriangularMesh(m_strPathRawObjFile,m_strPathObjFile);
}

