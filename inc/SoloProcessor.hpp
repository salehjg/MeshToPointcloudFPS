#pragma once

#include <iostream>
#include <string> 
#include "H5Cpp.h"

using namespace H5;

//http://www1.coe.neu.edu/~ctd/ISY240/CodingGuidelines.htm
class CSoloProcessor{
public:
    CSoloProcessor(string pathObjFile, string pathPcdFile, string pathHdf5File, int nSampledPoints);
    ~CSoloProcessor();
protected:
    int ProcessConvert2PCL();
    int ProcessSampleFPS();
    int SaveTheSampledPclToHDF5();
private:
    int m_iNumSampledPoints;
    string m_strPathObjFile, m_strPathHdf5File, m_strPathPcdFile;
    CMesh2Pcl *m_oMesh2Pcl;
}