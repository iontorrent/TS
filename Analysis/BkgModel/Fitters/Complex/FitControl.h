/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef FITCONTROL_H
#define FITCONTROL_H

#include <map>
#include <string>
#include "BkgFitStructures.h"

using namespace std;

class FitControl_t
{
  master_fit_type_table* bkg_model_fit_type;
  map<string, BkgFitMatrixPacker*> fittingMatrices;
  PartialDeriv_comp_list_item *PartialDeriv_comp_list;

  public:
    FitControl_t( master_fit_type_table *table );
    ~FitControl_t();
    void AllocPackers(float *tfptr, bool no_rdr_fit_starting_block,bool fitting_taue, int hydrogenModelType, int bead_flow_t, int npts, int flow_block_size);
    bool AddFitPacker(const char* fitName, int numFrames, int flow_block_size);
    BkgFitMatrixPacker* GetFitPacker(const string& fitName) const;

  private:
    void AllocWellPackers(int hydrogenModelType, int npts, int flow_block_size);
    void AllocRegionPackers(bool no_rdr_fit_starting_block, bool fitting_taue, int hydrogenModelType, int npts, int flow_block_size);
    void CombinatorialAllocationOfInit2AndFull(bool no_rdr_fit_starting_block, bool fitting_taue, int hydrogenModelType, int npts, int flow_block_size);
    void Delete();
};

#endif // FITCONTROL_H
