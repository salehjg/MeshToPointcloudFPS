#pragma once

#include <iostream>
#include <string>

extern int exportHdf5(
        const char *h5path, 
        float *data,
        int nPoints, int nDim);