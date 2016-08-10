/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved 
 *
 * BeadParamCubeClass.cpp
 *
 *  Created on: Sep 1, 2015
 *      Author: Jakob Siegel
 */


#include "PerBeadDataCubes.h"


#include "BeadParams.h"

#include "JobWrapper.h"
#include "SignalProcessingFitterQueue.h"

/////////////////////////////////////////////////
//perBeadParamCubeClass


//pBase_bkinfo => base pointer to array with  iP.getNumRegions() regions.
void perBeadParamCubeClass::translateHostToCube(BkgModelWorkInfo* pBase_bkinfo)
{
  assert ( this->memTypeHost() );
  assert(this->getDimZ() >= Bp_NUM_PARAMS);
  this->setRWStrideZ();
  ImgRegParams iP = getParams();
  for(size_t i=0; i < iP.getNumRegions(); i++)
  {
    WorkSet myJob(&pBase_bkinfo[i]);

    if(myJob.DataAvailalbe()){
      size_t regId = iP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      BeadParams * bP = myJob.getBeadParams();
      int numLBeads = myJob.getNumBeads();

      for(int b = 0; b < numLBeads; b++){
        this->setRWPtrRegion(regId,bP->x,bP->y);
        //Do not change order since device access is based on enum
        this->write(bP->Copies);
        this->write(bP->R);
        this->write(bP->dmult);
        this->write(bP->gain);
        this->write(bP->tau_adj);
        this->write(bP->phi);
        for(int p = 0; p < NUM_DM_PCA; p++) this->write(bP->pca_vals[p]);
        bP++;
      }
    }
  }
}

void perBeadParamCubeClass::init(BkgModelWorkInfo* pBase_bkinfo)
{
  // assert if known memory type
  assert(memTypeHost() || memTypeDevice());

  perBeadParamCubeClass * me = this;

  if(memTypeDevice()) //if device side buffer create host copy
    me = new perBeadParamCubeClass(*this,HostMem);

  me->translateHostToCube(pBase_bkinfo);

  if(memTypeDevice()){
    //copy and delete temp host buffer
    this->copy(*me);
    delete me;
  }
}


/////////////////////////////////////////////////
//perBeadPolyClonalCubeClass



void perBeadPolyClonalCubeClass::initHostRegion(BkgModelWorkInfo* pRegion_bkinfo, size_t regId)
{

  WorkSet myJob(pRegion_bkinfo);

  BeadParams * bP = myJob.getBeadParams();
  int numLBeads = myJob.getNumBeads();

  this->setRWStrideZ();

  assert(this->getDimZ() >= Poly_NUM_PARAMS);

  for(int b = 0; b < numLBeads; b++){
    this->setRWPtrRegion(regId,bP->x,bP->y);
    this->write(bP->my_state->ppf);
    this->write(bP->my_state->ssq);
    this->write(bP->my_state->key_norm);
    bP++;
  }
}

//pBase_bkinfo => base pointer to array with  iP.getNumRegions() regions.
void perBeadPolyClonalCubeClass::initHost(BkgModelWorkInfo* pBase_bkinfo)
{

  assert ( this->memTypeHost() );

  ImgRegParams iP = getParams();
  for(size_t i=0; i < iP.getNumRegions(); i++)
  {
    WorkSet myJob(&pBase_bkinfo[i]);
    if(myJob.DataAvailalbe()){
      size_t regId = iP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      initHostRegion(&pBase_bkinfo[i],regId);
    }
  }
}

void perBeadPolyClonalCubeClass::init(BkgModelWorkInfo* pBase_bkinfo)
{
  // assert if known memory type
  assert(memTypeHost() || memTypeDevice());

  if(memTypeHost())
  {
    initHost(pBase_bkinfo);
  }
  else
  {
    perBeadPolyClonalCubeClass tmpHostPolyClonalCube(getParams(), HostMem);
    tmpHostPolyClonalCube.initHost(pBase_bkinfo);
    this->copy(tmpHostPolyClonalCube);
  }
}

void perBeadPolyClonalCubeClass::reinjectHostStructures(BkgModelWorkInfo* pBase_bkinfo)
{
  assert(memTypeHost() || memTypeDevice());

  perBeadPolyClonalCubeClass * me = this;

  if(memTypeDevice())
    me = new perBeadPolyClonalCubeClass(*this, HostMem);

  ImgRegParams iP = getParams();
  me->setRWStrideZ();
  assert(me->getDimZ() >= Poly_NUM_PARAMS);

  for(size_t i=0; i < iP.getNumRegions(); i++)
  {
    WorkSet myJob(&pBase_bkinfo[i]);
    BeadParams * bP = myJob.getBeadParams();
    int numLBeads = myJob.getNumBeads();
    if(myJob.DataAvailalbe()){
      size_t regId = iP.getRegId(myJob.getRegCol(), myJob.getRegRow());
      for(int b = 0; b < numLBeads; b++){
        me->setRWPtrRegion(regId,bP->x,bP->y);
        bP->my_state->ppf = me->read();
        bP->my_state->ssq = me->read();
        bP->my_state->key_norm = me->read();
        bP++;
      }
      me->initHostRegion(&pBase_bkinfo[i],regId);
    }
  }
  if(me != NULL && me != this)
    delete me;
}


/////////////////////////////////////////////////
//perBeadT0CubeClass
void perBeadT0CubeClass::init(BkgModelWorkInfo* pBkinfo)
{
  copyIn( &(*(pBkinfo->smooth_t0_est))[0]);
}

/////////////////////////////////////////////////
//perBeadStateMaskClass


void perBeadStateMaskClass::initHostRegion(BkgModelWorkInfo* pRegion_bkinfo, size_t regId)
{
  WorkSet myJob(pRegion_bkinfo);

  BeadParams * bP = myJob.getBeadParams();
  int numLBeads = myJob.getNumBeads();

  for(int b = 0; b < numLBeads; b++){
    bead_state * Bs = bP->my_state;
    unsigned short maskValue = 0;
    maskValue |= (Bs->bad_read)?(BkgMaskBadRead):(0);
    maskValue |= (!Bs->clonal_read)?(BkgMaskPolyClonal):(0);
    maskValue |= (Bs->corrupt)?(BkgMaskCorrupt):(0);
    maskValue |= (Bs->pinned)?(BkgMaskPinned):(0);
    maskValue |= (Bs->random_samp)?(BkgMaskRandomSample):(0);
    maskValue |= (pRegion_bkinfo->bkgObj->region_data->my_beads.sampled[b])?(BkgMaskRegionalSampled):(0);
    maskValue |= (pRegion_bkinfo->bkgObj->region_data->my_beads.high_quality[b])?(BkgMaskHighQaulity):(0);

    putAtReg(maskValue, regId, bP->x, bP->y);

    bP++;
  }
}
void perBeadStateMaskClass::initHost(BkgModelWorkInfo* pBase_bkinfo)
{

  assert ( this->memTypeHost() );

  ImgRegParams iP = getParams();
  this->memSet(0);

  for(size_t i=0; i< iP.getNumRegions(); i++){
    WorkSet myJob(&pBase_bkinfo[i]);
    if(myJob.DataAvailalbe()){
      size_t regId = iP.getRegId(myJob.getRegCol(),myJob.getRegRow());
      initHostRegion(&pBase_bkinfo[i],regId);
    }
  }

}


void perBeadStateMaskClass::reinjectHostStructures(BkgModelWorkInfo* pBase_bkinfo)
{
  assert ( this->memTypeHost() );

  ImgRegParams iP = getParams();
  for(size_t i=0; i< iP.getNumRegions(); i++){

    WorkSet myJob(&pBase_bkinfo[i]);
    if(myJob.DataAvailalbe()){

      size_t regId = iP.getRegId(myJob.getRegCol(),myJob.getRegRow());
      BeadParams * bP = myJob.getBeadParams();
      int numLBeads = myJob.getNumBeads();

      for(int b = 0; b < numLBeads; b++){

        bead_state * Bs = bP->my_state;
        unsigned short maskValue = getAtReg(regId, bP->x, bP->y);
        Bs->bad_read = ( maskValue & BkgMaskBadRead );
        Bs->clonal_read = !(maskValue & BkgMaskPolyClonal);
        Bs->corrupt  = ( maskValue & BkgMaskCorrupt);
        Bs->pinned = ( maskValue & BkgMaskPinned);
        Bs->random_samp = ( maskValue & BkgMaskRandomSample);
        pBase_bkinfo[i].bkgObj->region_data->my_beads.sampled[b] = ( maskValue & BkgMaskRegionalSampled);
        pBase_bkinfo[i].bkgObj->region_data->my_beads.high_quality[b] = ( maskValue & BkgMaskHighQaulity);
        bP++;

      }
    }
  }
}


void perBeadStateMaskClass::init(BkgModelWorkInfo* pBase_bkinfo)
{
  // assert if known memory type
  assert(memTypeHost() || memTypeDevice());

  if(memTypeHost())
  {
    initHost(pBase_bkinfo);
  }
  else
  {
    perBeadStateMaskClass tmpHostBeadStateMask(getParams(), HostMem);
    tmpHostBeadStateMask.initHost(pBase_bkinfo);
    this->copy(tmpHostBeadStateMask);
  }
}


/////////////////////////////////////////////////
//perBeadBfMaskClass


void perBeadBfMaskClass::init(BkgModelWorkInfo* pBkinfo)
{
  assert(memTypeHost() || memTypeDevice());

  if(memTypeHost())
  {
    wrappPtr(&pBkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);
  }
  else
  {
    copyIn(&pBkinfo->bkgObj->GetGlobalStage().bfmask->mask[0]);
  }
}


/////////////////////////////////////////////////
//perBeadTraceCubeClass

void perBeadTraceCubeClass::init(BkgModelWorkInfo* pBkinfo)
{
  assert(memTypeHost() || memTypeDevice());

  Image * pImg = pBkinfo->img;

  if(memTypeHost())
  {
    wrappPtr(pImg->GetImage()->image); //wrap raw image
  }
  else
  {
    //create perBeadTracCube with uncompressed frames for copy to device
    //use temporary host cube instead of copying directly from host pointer
    //since compressed frames might be < or > than VFCompressed traces
    //checking and correct copying therefore is done by cube internals.
    perBeadTraceCubeClass tmpHostTraceCube(NULL,this->getParams(),pImg->GetMaxFrames(), HostMem);
    tmpHostTraceCube.init(pBkinfo);
    copy(tmpHostTraceCube);
  }

}





