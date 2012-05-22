/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FITCONTROL_H
#define FITCONTROL_H

#include "BkgFitStructures.h"


// this should be a map rather than individual pointers
// it is 2011 and the STL has been in existencde for some time
class FitControl_t
{
  public:
    const BkgFitStructures fitParams;
    // helper class to manipulate PartialDerivriv terms and solve the non-linear
    PartialDeriv_comp_list_item *PartialDeriv_comp_list;
    // regression calculation
    BkgFitMatrixPacker *FitAmpl;
    BkgFitMatrixPacker *FitInitial;
    BkgFitMatrixPacker *FitR;
    BkgFitMatrixPacker *FitPostKey;
    BkgFitMatrixPacker *FitKnownSequence;
    BkgFitMatrixPacker *FitWellWithBkg;
    BkgFitMatrixPacker *FitWellMinimal;
    BkgFitMatrixPacker *DontFitWells;

    // used for fitting tau, NucModifyRatio and oR to the whole region, not just individual wells
    BkgFitMatrixPacker *FitRegionInit1;
    BkgFitMatrixPacker *FitRegionInit2;
    BkgFitMatrixPacker *FitRegionFull;
    BkgFitMatrixPacker *FitRegionSlimErr;
    BkgFitMatrixPacker *FitRegionSlim;
    BkgFitMatrixPacker *DontFitRegion;

    FitControl_t();
    void Delete();
    ~FitControl_t();
    void AllocPackers(float *tfptr, bool no_rdr_fit_starting_block, int bead_flow_t, int npts);
};




#endif // FITCONTROL_H