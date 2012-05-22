/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "FitControl.h"



FitControl_t::FitControl_t()
{
    PartialDeriv_comp_list = NULL;
    FitAmpl     = NULL;
    FitInitial = NULL;
    FitR = NULL;
    FitPostKey   = NULL;
    DontFitWells = NULL;
    FitRegionInit1= NULL;
    FitRegionInit2= NULL;
    FitRegionFull= NULL;
    FitRegionSlim= NULL;
    FitRegionSlimErr = NULL;
    FitKnownSequence=NULL;
    FitWellWithBkg = NULL;
    FitWellMinimal = NULL;
    DontFitRegion = NULL;
}

void FitControl_t::Delete()
{
    if (FitWellMinimal != NULL) delete FitWellMinimal;
    if (FitWellWithBkg != NULL) delete FitWellWithBkg;
    if (FitKnownSequence != NULL) delete FitKnownSequence;
    if (FitRegionInit1 != NULL) delete FitRegionInit1;
    if (FitRegionInit2 != NULL) delete FitRegionInit2;
    if (FitRegionFull != NULL) delete FitRegionFull;
    if (FitRegionSlim != NULL) delete FitRegionSlim;
    if (FitRegionSlimErr != NULL) delete FitRegionSlimErr;
    if (FitR != NULL) delete FitR;
    if (FitInitial != NULL) delete FitInitial;
    if (FitAmpl != NULL) delete FitAmpl;
    if (FitPostKey != NULL) delete FitPostKey;
    if (DontFitWells !=NULL) delete DontFitWells;  // should never happen!!!
    if (DontFitRegion !=NULL) delete DontFitRegion; // should never happen!!!

    if (PartialDeriv_comp_list != NULL) delete [] PartialDeriv_comp_list;
}

FitControl_t::~FitControl_t()
{
  Delete();
}


void FitControl_t::AllocPackers(float *tfptr, bool no_rdr_fit_starting_block, int bead_flow_t, int npts)
{
    PartialDeriv_comp_list = new PartialDeriv_comp_list_item[fitParams.NumSteps];
    for (int i=0;i<fitParams.NumSteps;i++)
    {
        fitParams.Steps[i].ptr = tfptr;
        PartialDeriv_comp_list[i].addr = tfptr;
        PartialDeriv_comp_list[i].comp = (PartialDerivComponent) fitParams.Steps[i].PartialDerivMask;
        tfptr += bead_flow_t;
    }
    int comp_list_len = fitParams.NumSteps;

    if (FitAmpl) {
        delete FitAmpl;
    }
    if (FitR) {
        delete FitR;
    }
    if (FitInitial) {
        delete FitInitial;
    }
    if (FitPostKey) {
        delete FitPostKey;
    }
    if (FitKnownSequence) {
        delete FitKnownSequence;
    }
    if (FitWellWithBkg) {
        delete FitWellWithBkg;
    }
    if (FitWellMinimal) {
        delete FitWellMinimal;
    }
    if (FitRegionInit1) {
        delete FitRegionInit1;
    }

    FitAmpl = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitAmpl"),PartialDeriv_comp_list,comp_list_len);
    FitR = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitR"),PartialDeriv_comp_list,comp_list_len);
    FitInitial = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitInitial"),PartialDeriv_comp_list,comp_list_len);
    FitPostKey = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitPostKey"),PartialDeriv_comp_list,comp_list_len);
    FitKnownSequence = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitKnownSequence"),PartialDeriv_comp_list,comp_list_len);
    FitWellWithBkg = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitWellWithBkg"),PartialDeriv_comp_list,comp_list_len);
    FitWellMinimal = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitWellMinimal"),PartialDeriv_comp_list,comp_list_len);

    FitRegionInit1 = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitRegionInit1"),PartialDeriv_comp_list,comp_list_len);

    if (no_rdr_fit_starting_block)
    {
        if (FitRegionInit2) {
            delete FitRegionInit2;
        }
        if (FitRegionFull) {
            delete FitRegionFull;
        }
        FitRegionInit2 = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitRegionInit2NoRDR"),PartialDeriv_comp_list,comp_list_len);
        FitRegionFull = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitRegionFullNoRDR"),PartialDeriv_comp_list,comp_list_len);
    }
    else
    {
        if (FitRegionInit2) {
            delete FitRegionInit2;
        }
        if (FitRegionFull) {
            delete FitRegionFull;
        }
        FitRegionInit2 = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitRegionInit2"),PartialDeriv_comp_list,comp_list_len);
        FitRegionFull = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitRegionFull"),PartialDeriv_comp_list,comp_list_len);
    }

    FitRegionSlim = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitRegionSlim"),PartialDeriv_comp_list,comp_list_len);
    FitRegionSlimErr = new BkgFitMatrixPacker(npts,*GetFitInstructionsByName("FitRegionSlimErr"),PartialDeriv_comp_list,comp_list_len);
}