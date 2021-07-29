#include <vtkCellArray.h>

#ifdef VTK_CELL_ARRAY_V2
  using vtkCellPtsPtr = vtkIdType const*;
#else
  using vtkCellPtsPtr = vtkIdType*;
#endif