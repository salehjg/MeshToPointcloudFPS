#include "SoloProcessor.hpp"

/**
 * @brief Class Constructor
 * @details The CUDA device must be already initialized outside, 
 *          this is to make sure that there wont be extra delays when doing batch operations.
 * 
 * @param pathObjFile The path to the input mesh file(*.obj) to be read by PCL library.
 * @param pathPcdFile The path to the optional output pointcloud file(*.pcd), will be 
 *                    generated if the value is not empty("").
 * @param pathHdf5File The path to the output FPS sampled pointcloud file (*.h5).
 * @param nSampledPoints The target number of points to be sampled from the raw pointcloud by 
 *                       FPS algorithm.
 */
CSoloProcessor::CSoloProcessor(
        string pathObjFile, 
        string pathPcdFile, 
        string pathHdf5File, 
        int nSampledPoints){

    m_iNumSampledPoints = nSampledPoints;
    m_strPathObjFile = pathObjFile;
    m_strPathPcdFile = pathPcdFile;
    m_strPathHdf5File = pathHdf5File;

    m_oMesh2Pcl = new CMesh2Pcl(m_strPathObjFile, m_strPathPcdFile)
}

/**
 * @brief Class Destructor
 * @details The host side and the device side buffers along with class instances will be 
 *          released here manually.
 * @return .
 */
CSoloProcessor::~CSoloProcessor(){

}

int CSoloProcessor::ProcessConvert2PCL(){
    return m_oMesh2Pcl->Convert();
}

int CSoloProcessor::ProcessSampleFPS(){
    // 1. create the host side buff
    // 2. clone the pcl to the buff
    // 3. transfer the buf to the gpu mem
    // 4. launch the FPS kernel on gpu
    // 5. transfer back the results (indices) to the host
    // 6. gather the indices from the buff
    // 7. release the gpu side buffers
    // 8. release the host side buffers
    // 9. keep the final result
    // 10. return the integer status 
    return 0;
}

int CSoloProcessor::SaveTheSampledPclToHDF5(){
    
}
