//
// Created by saleh on 8/11/21.
//

#include "CPureTriMesh.h"

bool CPureTriMesh::hasDataMask(const int mask, const int maskToBeTested) {
    return ((mask & maskToBeTested)!= 0);
}

void CPureTriMesh::clearDataMask(CPureTriMesh::MyMesh &cm, int &mask, int unneededDataMask) {
    if( ( (unneededDataMask & MM_VERTFACETOPO)!=0)	&& hasDataMask(mask,MM_VERTFACETOPO)) {cm.face.DisableVFAdjacency();
        cm.vert.DisableVFAdjacency(); }
    if( ( (unneededDataMask & MM_FACEFACETOPO)!=0)	&& hasDataMask(mask,MM_FACEFACETOPO))	cm.face.DisableFFAdjacency();

    if( ( (unneededDataMask & MM_WEDGTEXCOORD)!=0)	&& hasDataMask(mask,MM_WEDGTEXCOORD)) 	cm.face.DisableWedgeTexCoord();
    if( ( (unneededDataMask & MM_FACECOLOR)!=0)			&& hasDataMask(mask,MM_FACECOLOR))			cm.face.DisableColor();
    if( ( (unneededDataMask & MM_FACEQUALITY)!=0)		&& hasDataMask(mask,MM_FACEQUALITY))		cm.face.DisableQuality();
    if( ( (unneededDataMask & MM_FACEMARK)!=0)			&& hasDataMask(mask,MM_FACEMARK))			cm.face.DisableMark();
    if( ( (unneededDataMask & MM_VERTMARK)!=0)			&& hasDataMask(mask,MM_VERTMARK))			cm.vert.DisableMark();
    if( ( (unneededDataMask & MM_VERTCURV)!=0)			&& hasDataMask(mask,MM_VERTCURV))			cm.vert.DisableCurvature();
    if( ( (unneededDataMask & MM_VERTCURVDIR)!=0)		&& hasDataMask(mask,MM_VERTCURVDIR))		cm.vert.DisableCurvatureDir();
    if( ( (unneededDataMask & MM_VERTRADIUS)!=0)		&& hasDataMask(mask,MM_VERTRADIUS))		cm.vert.DisableRadius();
    if( ( (unneededDataMask & MM_VERTTEXCOORD)!=0)	&& hasDataMask(mask,MM_VERTTEXCOORD))	cm.vert.DisableTexCoord();

    mask = mask & (~unneededDataMask);
}

int CPureTriMesh::MakeObjFilePureTriangularMesh(const string& inObjFilePath, const string& outObjFilePath) {
    MyMesh m;
    int loadMask;
    if(vcg::tri::io::ImporterOBJ<MyMesh>::ErrorCritical(vcg::tri::io::ImporterOBJ<MyMesh>::Open(m, inObjFilePath.c_str(),loadMask))){
        return 1;
    }

    vcg::tri::BitQuadCreation<MyMesh>::MakeBitTriOnly(m);
    clearDataMask(m, loadMask, MM_POLYGONAL);

    if(vcg::tri::io::ExporterOBJ<MyMesh>::Save(
            m,
            outObjFilePath.c_str(),
            vcg::tri::io::Mask::IOM_WEDGTEXCOORD,
            false,
            0)
            !=0){
        return 2;
    }
    return 0;
}
