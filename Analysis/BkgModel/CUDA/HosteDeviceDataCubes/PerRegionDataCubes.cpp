/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * PerRegionDataCubes.cpp
 *
 *  Created on: Sep 22, 2015
 *      Author: Jakob Siegel
 */



#include "PerRegionDataCubes.h"


#include "RegionParams.h"

#include "JobWrapper.h"
#include "SignalProcessingFitterQueue.h"


/////////////////////////////////////////////////
//RegionalParamCubeClass

//pBase_bkinfo => base pointer to array with  iP.getNumRegions() regions.
void RegionalParamCubeClass::translateHostToCube(BkgModelWorkInfo* pBase_bkinfo)
{
  assert ( this->memTypeHost() );

  ImgRegParams iP = getParams();
  for(size_t i=0; i < iP.getNumRegions(); i++)
  {
    WorkSet myJob(&pBase_bkinfo[i]);
    if(myJob.DataAvailalbe()){
      size_t regId = iP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      reg_params * rp = &(pRegion_bkinfo->bkgObj->region_data->my_regions.rp);
      PerFlowParamsRegion & ref = this->refAtReg(regId);
      ref.setCopyDrift(rp->CopyDrift);
      ref.setDarkness(rp->darkness[0]);
      ref.setRatioDrift(rp->RatioDrift);
      ref.setSigma(*(rp->AccessSigma()));
      ref.setFineStart(region_bkinfo->bkgObj->region_data->my_regions.cache_step.i_start_fine_step[0]);
      ref.setCoarseStart(region_bkinfo->bkgObj->region_data->my_regions.cache_step.i_start_coarse_step[0]);
      ref.setTMidNuc(rp->AccessTMidNuc()[0]);
      ref.setTMidNucShift(rp->nuc_shape.t_mid_nuc_shift_per_flow[0]);
      ref.setTshift(rp->tshift);
    }
  }
}

void RegionalParamCubeClass::translateCubeToHost(BkgModelWorkInfo* pBase_bkinfo)
{
  assert ( this->memTypeHost() );

  ImgRegParams iP = getParams();
  for(size_t i=0; i < iP.getNumRegions(); i++)
  {
    WorkSet myJob(&pBase_bkinfo[i]);
    if(myJob.DataAvailalbe()){
      size_t regId = iP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      reg_params * rp = &(pRegion_bkinfo->bkgObj->region_data->my_regions.rp);
      PerFlowParamsRegion & ref = this->refAtReg(regId);

      rp->CopyDrift = ref.getCopyDrift();
      rp->darkness[0] = ref.getDarkness();
      rp->RatioDrift = ref.getRatioDrift();
      *(rp->AccessSigma()) = ref.getSigma();
      region_bkinfo->bkgObj->region_data->my_regions.cache_step.i_start_fine_step[0] = ref.getStart();
      rp->AccessTMidNuc()[0] = ref.getTMidNuc();
      rp->nuc_shape.t_mid_nuc_shift_per_flow[0] = ref.getTMidNucShift();
      rp->tshift = ref.getTshift();
    }
  }
}


void RegionalParamCubeClass::init(BkgModelWorkInfo* pBase_bkinfo)
{
  // assert if known memory type
   assert(memTypeHost() || memTypeDevice());

   RegionalParamCubeClass * me = this;

   if(memTypeDevice()) //if device side buffer create host copy
     me = new RegionalParamCubeClass(*this,HostMem);

   me->translateHostToCube(pBase_bkinfo);

   if(memTypeDevice()){
     //copy and delete temp host buffer
     this->copy(*me);
     delete me;
   }
}

void RegionalParamCubeClass::reinjectHostStructures(BkgModelWorkInfo* pBase_bkinfo)
{
  // assert if known memory type
   assert(memTypeHost() || memTypeDevice());

   RegionalParamCubeClass * me = this;

   if(memTypeDevice())
     me = new RegionalParamCubeClass(*this,HostMem);

   me->translateCubeToHost(pBase_bkinfo);

   if(memTypeDevice()){
     //delete temp host buffer
     delete me;
   }

}


