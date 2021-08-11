#pragma once

#include <string>
#include<vcg/complex/complex.h>
#include<wrap/io_trimesh/import_obj.h>
#include<wrap/io_trimesh/export_obj.h>
#include <vcg/complex/algorithms/bitquad_creation.h>

using namespace std;

class CPureTriMesh {
public:
    int MakeObjFilePureTriangularMesh(const string& inObjFilePath, const string& outObjFilePath);
private:
    // Forward declarations needed for creating the used types
    class CVertexO;
    class CEdgeO;
    class CFaceO;

    // Declaration of the semantic of the used types
    class CUsedTypesO: public vcg::UsedTypes < vcg::Use<CVertexO>::AsVertexType,
            vcg::Use<CEdgeO   >::AsEdgeType,
            vcg::Use<CFaceO  >::AsFaceType >{};


    // The Main Vertex Class
    // Most of the attributes are optional and must be enabled before use.
    // Each vertex needs 40 byte, on 32bit arch. and 44 byte on 64bit arch.

    class CVertexO  : public vcg::Vertex< CUsedTypesO,
            vcg::vertex::InfoOcf,           /*  4b */
            vcg::vertex::Coord3f,           /* 12b */
            vcg::vertex::BitFlags,          /*  4b */
            vcg::vertex::Normal3f,          /* 12b */
            vcg::vertex::Qualityf,          /*  4b */
            vcg::vertex::Color4b,           /*  4b */
            vcg::vertex::VFAdjOcf,          /*  0b */
            vcg::vertex::MarkOcf,           /*  0b */
            vcg::vertex::TexCoordfOcf,      /*  0b */
            vcg::vertex::CurvaturefOcf,     /*  0b */
            vcg::vertex::CurvatureDirfOcf,  /*  0b */
            vcg::vertex::RadiusfOcf         /*  0b */
    >{
    };

    // The Main Edge Class
    // Currently it does not contains anything.
    class CEdgeO : public vcg::Edge<CUsedTypesO,
            vcg::edge::BitFlags,          /*  4b */
            vcg::edge::EVAdj,
            vcg::edge::EEAdj
    >{
    };

    // Each face needs 32 byte, on 32bit arch. and 48 byte on 64bit arch.
    class CFaceO    : public vcg::Face<  CUsedTypesO,
            vcg::face::InfoOcf,              /* 4b */
            vcg::face::VertexRef,            /*12b */
            vcg::face::BitFlags,             /* 4b */
            vcg::face::Normal3f,             /*12b */
            vcg::face::QualityfOcf,          /* 0b */
            vcg::face::MarkOcf,              /* 0b */
            vcg::face::Color4bOcf,           /* 0b */
            vcg::face::FFAdjOcf,             /* 0b */
            vcg::face::VFAdjOcf,             /* 0b */
            vcg::face::WedgeTexCoord2f     /* 0b */
    > {};

    class MyMesh: public vcg::tri::TriMesh< vcg::vertex::vector_ocf<CVertexO>, vcg::face::vector_ocf<CFaceO> >{};

    /* ADOPTED FROM MESHLAB'S SOURCE CODE
	This enum specify the various simplex components
	It is used in various parts of the framework:
	- to know what elements are currently active and therefore can be saved on a file
	- to know what elements are required by a filter and therefore should be made ready before starting the filter (e.g. if a
	- to know what elements are changed by a filter and therefore should be saved/restored in case of dynamic filters with a preview
	*/
    enum MeshElement{
        MM_NONE           = 0x00000000,
        MM_VERTCOORD      = 0x00000001,
        MM_VERTNORMAL     = 0x00000002,
        MM_VERTFLAG       = 0x00000004,
        MM_VERTCOLOR      = 0x00000008,
        MM_VERTQUALITY    = 0x00000010,
        MM_VERTMARK       = 0x00000020,
        MM_VERTFACETOPO   = 0x00000040,
        MM_VERTCURV	      = 0x00000080,
        MM_VERTCURVDIR    = 0x00000100,
        MM_VERTRADIUS     = 0x00000200,
        MM_VERTTEXCOORD   = 0x00000400,
        MM_VERTNUMBER     = 0x00000800,

        MM_FACEVERT       = 0x00001000,
        MM_FACENORMAL     = 0x00002000,
        MM_FACEFLAG       = 0x00004000,
        MM_FACECOLOR      = 0x00008000,
        MM_FACEQUALITY    = 0x00010000,
        MM_FACEMARK       = 0x00020000,
        MM_FACEFACETOPO   = 0x00040000,
        MM_FACENUMBER     = 0x00080000,
        MM_FACECURVDIR    = 0x00100000,

        MM_WEDGTEXCOORD   = 0x00200000,
        MM_WEDGNORMAL     = 0x00400000,
        MM_WEDGCOLOR      = 0x00800000,

        // 	Selection
        MM_VERTFLAGSELECT = 0x01000000,
        MM_FACEFLAGSELECT = 0x02000000,

        // Per Mesh Stuff....
        MM_CAMERA         = 0x08000000,
        MM_TRANSFMATRIX   = 0x10000000,
        MM_COLOR          = 0x20000000,
        MM_POLYGONAL      = 0x40000000,

        // unknown - will raise exceptions, to be avoided, here just for compatibility
        MM_UNKNOWN        = 0x80000000,

        // geometry change (for filters that remove stuff or modify geometry or topology, but not touch face/vertex color or face/vertex quality)
        MM_GEOMETRY_AND_TOPOLOGY_CHANGE = 0x431e7be7,

        // everything - dangerous, will add unwanted data to layer (e.g. if you use MM_ALL it could means that it could add even color or quality)
        MM_ALL            = 0xffffffff
    };

    bool hasDataMask(const int mask, const int maskToBeTested);
    void clearDataMask(MyMesh &cm, int &mask, int unneededDataMask);
};





