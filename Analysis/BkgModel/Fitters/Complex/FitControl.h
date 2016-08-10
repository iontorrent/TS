/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FITCONTROL_H
#define FITCONTROL_H

#include "BkgFitStructures.h"


// this should be a map rather than individual pointers
// it is 2011 and the STL has been in existencde for some time
class FitControl_t
{
    master_fit_type_table* bkg_model_fit_type;
  public:
    // helper class to manipulate PartialDerivriv terms and solve the non-linear
    PartialDeriv_comp_list_item *PartialDeriv_comp_list;
    // regression calculation
    BkgFitMatrixPacker *FitWellAmpl;
    BkgFitMatrixPacker *FitWellAmplBuffering;
    BkgFitMatrixPacker *FitWellPostKey;
    BkgFitMatrixPacker *FitWellAll; // includes kmult
    BkgFitMatrixPacker *DontFitWells;

    // used for fitting tau, NucModifyRatio and oR to the whole region, not just individual wells
    BkgFitMatrixPacker *FitRegionTmidnucPlus;
    BkgFitMatrixPacker *FitRegionInit2;
    BkgFitMatrixPacker *FitRegionFull;
    BkgFitMatrixPacker *FitRegionDarkness;
    BkgFitMatrixPacker *FitRegionTimeVarying;
    BkgFitMatrixPacker *DontFitRegion;

    FitControl_t( master_fit_type_table *table );
    void Delete();
    void DeleteWellPackers();
    void DeleteRegionPackers();
    void NullWellPackers();
    void NullRegionPackers();
    ~FitControl_t();
    void AllocWellPackers(int hydrogenModelType, int npts, int flow_block_size);
    void AllocRegionPackers(bool no_rdr_fit_starting_block, bool fitting_taue, int hydrogenModelType, int npts, int flow_block_size);
    void CombinatorialAllocationOfInit2AndFull(bool no_rdr_fit_starting_block, bool fitting_taue, int hydrogenModelType, int npts, int flow_block_size);
    void AllocPackers(float *tfptr, bool no_rdr_fit_starting_block,bool fitting_taue, int hydrogenModelType, int bead_flow_t, int npts, int flow_block_size);
};




#endif // FITCONTROL_H
