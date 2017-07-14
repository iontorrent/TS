/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "FitControl.h"



FitControl_t::FitControl_t( master_fit_type_table *table ) :
  bkg_model_fit_type( table )
{
  PartialDeriv_comp_list = NULL;
  // well
  //NullWellPackers();
  // region
  //NullRegionPackers();
}

/*
void FitControl_t::NullWellPackers()
{
  FitWellAmpl     = NULL;
  FitWellAmplBuffering = NULL;
  FitWellPostKey   = NULL;
  DontFitWells = NULL;
  FitWellAll = NULL;
}
*/
/*void FitControl_t::NullRegionPackers()
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
  if ( FitWellAll!=NULL) delete FitWellAll;
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
*/

void FitControl_t::Delete()
{
  //DeleteWellPackers();

  //DeleteRegionPackers();

  for(auto it = fittingMatrices.begin(); it!= fittingMatrices.end(); ++it) {
    delete it->second;
  }
  fittingMatrices.clear();

  if ( PartialDeriv_comp_list != NULL ) delete [] PartialDeriv_comp_list;
}

FitControl_t::~FitControl_t()
{
  Delete();
}

void FitControl_t::AllocWellPackers ( int hydrogenModelType, int npts, int flow_block_size )
{
  //DeleteWellPackers();

  //int comp_list_len = BkgFitStructures::NumSteps;

  AddFitPacker("FitWellAmpl", npts, flow_block_size);
  //FitWellAmpl = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitWellAmpl" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
  //FitWellAmplBuffering = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitWellAmplBuffering" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
  AddFitPacker("FitWellAmplBuffering", npts, flow_block_size);

  switch( hydrogenModelType ){
    case 0:
    default:
        //FitWellPostKey = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitWellPostKey" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
        AddFitPacker("FitWellPostKey", npts, flow_block_size);
        break;
    case 1:
    case 2:
    case 3:
        //FitWellPostKey = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitWellPostKeyNoDmult" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
        AddFitPacker("FitWellPostKeyNoDmult", npts, flow_block_size);
        break;
  }
  //FitWellAll = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitWellAll" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
  AddFitPacker("FitWellAll", npts, flow_block_size);

}

void FitControl_t::CombinatorialAllocationOfInit2AndFull ( bool no_rdr_fit_starting_block,bool fitting_taue, int hydrogenModelType, int npts, int flow_block_size )
{
    int comp_list_len = BkgFitStructures::NumSteps;
    // combinatorial excess because we can't make up our minds
    switch( hydrogenModelType ){
    case 0:
    default:
        if ( no_rdr_fit_starting_block & ( !fitting_taue ) )
        {

            //FitRegionInit2 = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionInit2NoRDR" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionInit2NoRDR",npts,flow_block_size);
            //FitRegionFull = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionFullNoRDR" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionFullNoRDR",npts,flow_block_size);
        }

        if ( ( !no_rdr_fit_starting_block ) & ( !fitting_taue ) )
        {
            //FitRegionInit2 = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionInit2" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionInit2",npts,flow_block_size);
            //FitRegionFull = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionFull" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionFull",npts,flow_block_size);
        }

        if ( fitting_taue & ( !no_rdr_fit_starting_block ) )
        {
            //FitRegionInit2 = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionInit2TauE" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionInit2TauE",npts,flow_block_size);
            //FitRegionFull = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionFullTauE" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionFullTauE",npts,flow_block_size);
        }

        if ( fitting_taue & ( no_rdr_fit_starting_block ) )
        {
            //FitRegionInit2 = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionInit2TauENoRDR" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionInit2TauENoRDR",npts,flow_block_size);
            //FitRegionFull = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionFullTauENoRDR" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionFullTauENoRDR",npts,flow_block_size);
        }

        break;
    case 1:
    case 2:
    case 3:
        if ( fitting_taue & ( !no_rdr_fit_starting_block ) )
        {
            //FitRegionInit2 = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionInit2TauENoD" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionInit2TauENoD",npts,flow_block_size);
            //FitRegionFull = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionFullTauENoD" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionFullTauENoD",npts,flow_block_size);
        }

        if ( fitting_taue & ( no_rdr_fit_starting_block ) )
        {
            //FitRegionInit2 = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionInit2TauENoRDRNoD" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionInit2TauENoRDRNoD",npts,flow_block_size);
            //FitRegionFull = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionFullTauENoRDRNoD" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
            AddFitPacker("FitRegionFullTauENoRDRNoD",npts,flow_block_size);
        }
        break;
    }

    // end combinatorial excess
}

void FitControl_t::AllocRegionPackers ( bool no_rdr_fit_starting_block,bool fitting_taue,  int hydrogenModelType, int npts, int flow_block_size )
{
  int comp_list_len = BkgFitStructures::NumSteps;

  //DeleteRegionPackers();
  //FitRegionTmidnucPlus = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionTmidnucPlus" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
  AddFitPacker("FitRegionTmidnucPlus",npts,flow_block_size);

  CombinatorialAllocationOfInit2AndFull ( no_rdr_fit_starting_block,fitting_taue,hydrogenModelType,npts, flow_block_size );

  //FitRegionTimeVarying = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionTimeVarying" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
  AddFitPacker("FitRegionTimeVarying",npts,flow_block_size);
  //FitRegionDarkness = new BkgFitMatrixPacker ( npts,*bkg_model_fit_type->GetFitInstructionsByName ( "FitRegionDarkness" ),PartialDeriv_comp_list,comp_list_len, flow_block_size );
  AddFitPacker("FitRegionDarkness",npts,flow_block_size);
}

void FitControl_t::AllocPackers (float *tfptr, bool no_rdr_fit_starting_block, bool fitting_taue, int hydrogenModelType, int bead_flow_t, int npts, int flow_block_size )
{
  PartialDeriv_comp_list = new PartialDeriv_comp_list_item[BkgFitStructures::NumSteps];
  for ( int i=0;i<BkgFitStructures::NumSteps;i++ )
  {
    BkgFitStructures::Steps[i].ptr = tfptr;   // FIXME Really? We muck with a global structure?
    PartialDeriv_comp_list[i].addr = tfptr;
    PartialDeriv_comp_list[i].comp = ( PartialDerivComponent ) BkgFitStructures::Steps[i].PartialDerivMask;
    tfptr += bead_flow_t;
  }

  AllocWellPackers ( hydrogenModelType,  npts, flow_block_size );
  AllocRegionPackers ( no_rdr_fit_starting_block, fitting_taue, hydrogenModelType, npts , flow_block_size);
}

bool FitControl_t::AddFitPacker(
  const char* fitName,
  int numFrames,
  int flow_block_size)
{
  BkgFitMatrixPacker* packer = new BkgFitMatrixPacker (numFrames,*bkg_model_fit_type->GetFitInstructionsByName (fitName),PartialDeriv_comp_list,BkgFitStructures::NumSteps, flow_block_size);
  auto it = fittingMatrices.find(fitName);
  if (it != fittingMatrices.end()) {
    return false;
  }
  else {
   fittingMatrices[fitName] = packer;
  }
  
  return true;
}

BkgFitMatrixPacker* FitControl_t::GetFitPacker(const string& fitName) const {
  auto it =  fittingMatrices.find(fitName);
  if (it != fittingMatrices.end())
    return it->second;
  else 
    return NULL;
}
