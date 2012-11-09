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
    BkgFitMatrixPacker *FitWellAmpl;
    BkgFitMatrixPacker *FitWellAmplBuffering;
    BkgFitMatrixPacker *FitWellPostKey;
    BkgFitMatrixPacker *DontFitWells;

    // used for fitting tau, NucModifyRatio and oR to the whole region, not just individual wells
    BkgFitMatrixPacker *FitRegionTmidnucPlus;
    BkgFitMatrixPacker *FitRegionInit2;
    BkgFitMatrixPacker *FitRegionFull;
    BkgFitMatrixPacker *FitRegionDarkness;
    BkgFitMatrixPacker *FitRegionTimeVarying;
    BkgFitMatrixPacker *DontFitRegion;

    FitControl_t();
    void Delete();
    void DeleteWellPackers();
    void DeleteRegionPackers();
    void NullWellPackers();
    void NullRegionPackers();
    ~FitControl_t();
    void AllocWellPackers(int npts);
    void AllocRegionPackers(bool no_rdr_fit_starting_block,bool fitting_taue, int npts);
    void CombinatorialAllocationOfInit2AndFull(bool no_rdr_fit_starting_block,bool fitting_taue, int npts);
    void AllocPackers(float *tfptr, bool no_rdr_fit_starting_block,bool fitting_taue, int bead_flow_t, int npts);
};




#endif // FITCONTROL_H
