/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "BkgDataPointers.h"



// the functions called by the wrapper functions in BkgModel
void BkgDataPointers::copyCube_element(DataCube<float> *ptr, int x, int y, int j, float v)
{
    if (ptr != NULL) {
      ptr->At(x,y,j) = v; // At() not at() for Cube
    }

}


void BkgDataPointers::copyMatrix_element(arma::Mat<float> *ptr, int x, int j, float v)
{
    if (ptr != NULL) {
      ptr->at(x,j) =v; // at() not At() for matrix
    }
}


void BkgDataPointers::copyMatrix_element(arma::Mat<int> *ptr, int x, int j, int v)
{
    if (ptr != NULL) {
      ptr->at(x,j) =v; // at() not At() for matrix
    }
}


