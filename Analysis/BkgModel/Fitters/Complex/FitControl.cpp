/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "FitControl.h"



FitControl_t::FitControl_t()
{
  PartialDeriv_comp_list = NULL;
  // well
  NullWellPackers();
  // region
  NullRegionPackers();
}

void FitControl_t::NullWellPackers()
{
  FitWellAmpl     = NULL;
  FitWellAmplBuffering = NULL;
  FitWellPostKey   = NULL;
  DontFitWells = NULL;
}

void FitControl_t::NullRegionPackers()
{
  FitRegionTmidnucPlus= NULL;
  FitRegionInit2= NULL;
  FitRegionFull= NULL;
  FitRegionTimeVarying= NULL;
  FitRegionDarkness = NULL;
  DontFitRegion = NULL;
}

void FitControl_t::DeleteWellPackers()
{
  if ( FitWellAmplBuffering != NULL ) delete FitWellAmplBuffering;
  if ( FitWellAmpl != NULL ) delete FitWellAmpl;
  if ( FitWellPostKey != NULL ) delete FitWellPostKey;
  if ( DontFitWells !=NULL ) delete DontFitWells;  // should never happen!!!

  NullWellPackers();

}

void FitControl_t::DeleteRegionPackers()
{
  if ( FitRegionTmidnucPlus != NULL ) delete FitRegionTmidnucPlus;
  if ( FitRegionInit2 != NULL ) delete FitRegionInit2;
  if ( FitRegionFull != NULL ) delete FitRegionFull;
  if ( FitRegionTimeVarying != NULL ) delete FitRegionTimeVarying;
  if ( FitRegionDarkness != NULL ) delete FitRegionDarkness;
  if ( DontFitRegion !=NULL ) delete DontFitRegion; // should never happen!!!
  NullRegionPackers();
}

void FitControl_t::Delete()
{
  DeleteWellPackers();

  DeleteRegionPackers();


  if ( PartialDeriv_comp_list != NULL ) delete [] PartialDeriv_comp_list;
}

FitControl_t::~FitControl_t()
{
  Delete();
}

void FitControl_t::AllocWellPackers ( int npts )
{
  DeleteWellPackers();

  int comp_list_len = fitParams.NumSteps;

  FitWellAmpl = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitWellAmpl" ),PartialDeriv_comp_list,comp_list_len );
  FitWellAmplBuffering = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitWellAmplBuffering" ),PartialDeriv_comp_list,comp_list_len );
  FitWellPostKey = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitWellPostKey" ),PartialDeriv_comp_list,comp_list_len );
}

void FitControl_t::CombinatorialAllocationOfInit2AndFull ( bool no_rdr_fit_starting_block,bool fitting_taue,  int npts )
{
  int comp_list_len = fitParams.NumSteps;
  // combinatorial excess because we can't make up our minds
  if ( no_rdr_fit_starting_block & ( !fitting_taue ) )
  {

    FitRegionInit2 = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionInit2NoRDR" ),PartialDeriv_comp_list,comp_list_len );
    FitRegionFull = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionFullNoRDR" ),PartialDeriv_comp_list,comp_list_len );
  }

  if ( ( !no_rdr_fit_starting_block ) & ( !fitting_taue ) )
  {
    FitRegionInit2 = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionInit2" ),PartialDeriv_comp_list,comp_list_len );
    FitRegionFull = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionFull" ),PartialDeriv_comp_list,comp_list_len );
  }

  if ( fitting_taue & ( !no_rdr_fit_starting_block ) )
  {
    FitRegionInit2 = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionInit2TauE" ),PartialDeriv_comp_list,comp_list_len );
    FitRegionFull = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionFullTauE" ),PartialDeriv_comp_list,comp_list_len );
  }

  if ( fitting_taue & ( no_rdr_fit_starting_block ) )
  {
    FitRegionInit2 = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionInit2TauENoRDR" ),PartialDeriv_comp_list,comp_list_len );
    FitRegionFull = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionFullTauENoRDR" ),PartialDeriv_comp_list,comp_list_len );
  }
  // end combinatorial excess
}

void FitControl_t::AllocRegionPackers ( bool no_rdr_fit_starting_block,bool fitting_taue,  int npts )
{
  int comp_list_len = fitParams.NumSteps;

  DeleteRegionPackers();
  FitRegionTmidnucPlus = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionTmidnucPlus" ),PartialDeriv_comp_list,comp_list_len );

  CombinatorialAllocationOfInit2AndFull ( no_rdr_fit_starting_block,fitting_taue,npts );

  FitRegionTimeVarying = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionTimeVarying" ),PartialDeriv_comp_list,comp_list_len );
  FitRegionDarkness = new BkgFitMatrixPacker ( npts,*GetFitInstructionsByName ( "FitRegionDarkness" ),PartialDeriv_comp_list,comp_list_len );
}

void FitControl_t::AllocPackers ( float *tfptr, bool no_rdr_fit_starting_block,bool fitting_taue, int bead_flow_t, int npts )
{
  PartialDeriv_comp_list = new PartialDeriv_comp_list_item[fitParams.NumSteps];
  for ( int i=0;i<fitParams.NumSteps;i++ )
  {
    fitParams.Steps[i].ptr = tfptr;
    PartialDeriv_comp_list[i].addr = tfptr;
    PartialDeriv_comp_list[i].comp = ( PartialDerivComponent ) fitParams.Steps[i].PartialDerivMask;
    tfptr += bead_flow_t;
  }

  AllocWellPackers ( npts );
  AllocRegionPackers ( no_rdr_fit_starting_block,fitting_taue,npts );
}
