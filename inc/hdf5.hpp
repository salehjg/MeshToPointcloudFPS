#pragma once

#include <iostream>
#include <string>
#include "H5Cpp.h"

extern int exportHdf5(
        char *h5path, 
        pcl::PointCloud<pcl::PointXYZ> &cloud);