<p align="center">
	<img width="100%" src="https://github.com/./example.png">
</p>

# FpsCpu
`FpsCpu` is meant to be a simple `C++` based utility which converts mesh files (`*.obj`) to pointclouds. The data is down-sampled using Farthest Point Sampling algorithm (FPS). The results could be exported in various formats such as HDF5 (`*.h5`), PCL (`*.pcd`), and ...

# Development Status
- [x] CPU-only, Single-thread
- [ ] CPU-only, Multi-thread
- [ ] CPU-only, Multi-thread, Batch Operation Support
- [ ] GPU-CUDA
- [ ] GPU-OCL


# Dependencies
```
HDF5,  community/hdf5     in ArchLinux
PCL,   aur/pcl 1.11.1-2   in ArchLinux
CMake, extra/cmake        in ArchLinux
```

# Usage
As below:
```
$ FpsCpu -h
Usage: FpsCpu [options...]
Options:
    -i, --inputmesh        The path for the input mesh *.obj file. (Required)
    -o, --outputhdf5       The path for the output hdf5 *.h5 file with sampled point cloud. (Required)
    -n, --npoints          The target number of points per mesh input file (input.obj). (Required)
    -r, --rawpcd           The path for the optional output pcd *.pcd file with RAW point cloud.
    -p, --outputpcd        The path for the optional output pcd *.pcd file with sampled point cloud.
    -h, --help             Shows this page 

```

# Example
```
$ git clone https://github.com/salehjg/MeshToPointcloudFPS.git
$ cd MeshToPointcloudFPS
$ mkdir build
$ cd build
$ cmake ..
$ make -j8
$ ./FpsCpu -n 1024 -i ../data/tube.obj -o sampled.h5 -p sampled.pcd -r raw.pcd 
$ pcl_viewer raw.pcd
$ pcl_viewer sampled.pcd

```


# Credits
* The code for FPS algorithm is adopted from the CUDA kernel at [GitHub: charlesq34/pointnet2](https://github.com/charlesq34/pointnet2/blob/master/tf_ops/sampling/tf_sampling_g.cu).
* [GitHub: jamolnng/argparse](https://github.com/jamolnng/argparse)