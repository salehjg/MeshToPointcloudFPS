#include "hdf5.hpp" 
#include "H5Cpp.h"

using namespace H5;

int exportHdf5(
        const char *h5path, 
        float *data,
        int nPoints, int nDim){

    const H5std_string  FILE_NAME( h5path );
    try
    {
        /*
         * Turn off the auto-printing when failure occurs so that we can
         * handle the errors appropriately
         */
        Exception::dontPrint();
        /*
         * Create a new file using H5F_ACC_TRUNC access,
         * default file creation properties, and default file
         * access properties.
         */
        H5File file( FILE_NAME, H5F_ACC_TRUNC );
        /*
         * Define the size of the array and create the data space for fixed
         * size dataset.
         */
        hsize_t     dimsf[2];              // dataset dimensions
        dimsf[0] = nPoints;
        dimsf[1] = nDim; // x y z
        DataSpace dataspace( 2, dimsf ); //RANK=2
        /*
         * Define datatype for the data in the file.
         * We will store little endian INT numbers.
         */
        IntType datatype( PredType::NATIVE_FLOAT );
        datatype.setOrder( H5T_ORDER_LE );
        /*
         * Create a new dataset within the file using defined dataspace and
         * datatype and default dataset creation properties.
         */
        DataSet dataset = file.createDataSet( "data", datatype, dataspace );
        /*
         * Write the data to the dataset using default memory space, file
         * space, and transfer properties.
         */
        dataset.write( data, PredType::NATIVE_FLOAT );

    }  // end of try block
    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSpace operations
    catch( DataTypeIException error )
    {
        error.printErrorStack();
        return -1;
    }
    return 0;  // successfully terminated
}