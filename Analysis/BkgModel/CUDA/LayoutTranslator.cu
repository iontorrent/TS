/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <iostream>
#include "cuda_error.h"

#include "DeviceSymbolCopy.h"
#include "LayoutTranslator.h"
#include "SignalProcessingFitterQueue.h"
#include "JobWrapper.h"


#define TRANSLATE_DEBUG_OUTPUT 0



////////////////////////////////////
//Translator Functions


//Translate the FgBuffer into the Image cube
void TranslatorsFlowByFlow::TranslateFgBuffer_RegionToCube(  LayoutCubeWithRegions<short> & ImageCube,
    size_t numLBeads,
    size_t numFrames,
    size_t flowsPerBLock,
    FG_BUFFER_TYPE *fgPtr,
    BeadParams * bP,
    size_t regId)
{
  size_t x,y;
  //ImgRegParams ImageParams = ImageCube.getParams();
  //  ImageCube.SetValueRegion(0,regId);
  ImageCube.setRWStrideZ();
  for(size_t idx = 0; idx < numLBeads; idx++  ){
    x = bP->x;
    y = bP->y;
    ImageCube.setRWPtrRegion(regId,x,y);
    FG_BUFFER_TYPE * fgPtrFrames = fgPtr;
    for(size_t f = 0; f < numFrames; f++)
    {
      ImageCube.write(*fgPtrFrames);
      fgPtrFrames++;
    }
    // move to next bead
    fgPtr += numFrames*flowsPerBLock;
    bP++;
  }
}

void TranslatorsFlowByFlow::TranslateFgBuffer_CubeToRegion(  LayoutCubeWithRegions<short> & ImageCube,
    size_t numLBeads,
    size_t numFrames,
    size_t flowsPerBLock,
    FG_BUFFER_TYPE *fgPtr,
    BeadParams * bP,
    size_t regId)
{
  size_t x,y;
  //ImgRegParams ImageParams = ImageCube.getParams();

  ImageCube.setRWStrideZ();
  for(size_t idx = 0; idx < numLBeads; idx++  ){
    x = bP->x;
    y = bP->y;
    ImageCube.setRWPtrRegion(regId,x,y);
    FG_BUFFER_TYPE * fgPtrFrames = fgPtr;
    for(size_t f = 0; f < numFrames; f++)
    {
      *fgPtrFrames = ImageCube.read();
      fgPtrFrames++;
    }
    // move to next bead
    fgPtr += numFrames*flowsPerBLock;
    bP++;
  }
}


//Translate BeadParam struct into BeadParam Cube
void TranslatorsFlowByFlow::TranslateBeadParams_RegionToCube(LayoutCubeWithRegions<float> & BeadParamCube, void* bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

  BeadParams * bP = myJob.getBeadParams();
  int numLBeads = myJob.getNumBeads();

  BeadParamCube.setRWStrideZ();
  assert(BeadParamCube.getDimZ() >= Bp_NUM_PARAMS);

  for(int b = 0; b < numLBeads; b++){
    BeadParamCube.setRWPtrRegion(regId,bP->x,bP->y);
    BeadParamCube.write(bP->Copies);
    BeadParamCube.write(bP->R);
    BeadParamCube.write(bP->dmult);
    BeadParamCube.write(bP->gain);
    BeadParamCube.write(bP->tau_adj);
    BeadParamCube.write(bP->phi);
    for(int p = 0; p < NUM_DM_PCA; p++) BeadParamCube.write(bP->pca_vals[p]);
    bP++;
  }
}

void TranslatorsFlowByFlow::TranslateBeadParams_CubeToRegion(LayoutCubeWithRegions<float> & BeadParamCube, size_t numLBeads, BeadParams * bP, size_t regId)
{
  BeadParamCube.setRWStrideZ();

  assert(BeadParamCube.getDimZ() >= Bp_NUM_PARAMS);

  for(size_t b = 0; b < numLBeads; b++){
    BeadParamCube.setRWPtrRegion(regId,bP->x,bP->y);
    bP->Copies  = BeadParamCube.read();
    bP->R       = BeadParamCube.read();
    bP->dmult   = BeadParamCube.read();
    bP->gain    = BeadParamCube.read();
    bP->tau_adj = BeadParamCube.read();
    bP->phi     = BeadParamCube.read();
    for(int p = 0; p < NUM_DM_PCA; p++) bP->pca_vals[p] = BeadParamCube.read();
    bP++;
  }
}


//Translate Bead State into Bead State Cube
void TranslatorsFlowByFlow::TranslatePolyClonal_RegionToCube(LayoutCubeWithRegions<float> & PolyClonalCube, void* bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

  BeadParams * bP = myJob.getBeadParams();
  int numLBeads = myJob.getNumBeads();

  PolyClonalCube.setRWStrideZ();

  assert(PolyClonalCube.getDimZ() >= Poly_NUM_PARAMS);

  for(int b = 0; b < numLBeads; b++){
    PolyClonalCube.setRWPtrRegion(regId,bP->x,bP->y);
    PolyClonalCube.write(bP->my_state->ppf);
    PolyClonalCube.write(bP->my_state->ssq);
    PolyClonalCube.write(bP->my_state->key_norm);
    bP++;
  }
}


//Translate Bead State flags into BeadState Mask
void TranslatorsFlowByFlow::TranslateBeadStateMask_RegionToCube(  LayoutCubeWithRegions<unsigned short> & BkgModelMask, void* bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

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
    maskValue |= (info->bkgObj->region_data->my_beads.sampled[b])?(BkgMaskRegionalSampled):(0);
    maskValue |= (info->bkgObj->region_data->my_beads.high_quality[b])?(BkgMaskHighQaulity):(0);
    BkgModelMask.putAtReg(maskValue, regId, bP->x, bP->y);
    bP++;
  }
}


void TranslatorsFlowByFlow::TranslateBeadStateMask_CubeToRegion(  LayoutCubeWithRegions<unsigned short> & BkgModelMask, void* bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

  BeadParams * bP = myJob.getBeadParams();
  int numLBeads = myJob.getNumBeads();
  for(int b = 0; b < numLBeads; b++){
    unsigned short maskValue = 0;
    maskValue = BkgModelMask.getAtReg(regId, bP->x, bP->y);
    //sofar only Corrupt flag might get updated in kernel
   // bP->my_state->bad_read = (maskValue & BkgMaskBadRead);
    bP->my_state->corrupt = (maskValue & BkgMaskCorrupt);
    bP->my_state->clonal_read = (!(maskValue & BkgMaskPolyClonal));
  }
}


//translate Results from beadParams to Result Cube
void TranslatorsFlowByFlow::TranslateResults_RegionToCube(LayoutCubeWithRegions<float> & ResultCube, size_t numLBeads, size_t flowIdxInBlock, BeadParams * bP, size_t regId){
  ResultCube.setRWStrideZ();
  assert(ResultCube.getDimZ() >= Result_NUM_PARAMS);

  for(size_t b = 0; b < numLBeads; b++){
    ResultCube.setRWPtrRegion(regId,bP->x,bP->y,ResultAmpl);
    ResultCube.write(bP->Ampl[flowIdxInBlock]);
    ResultCube.write(bP->kmult[flowIdxInBlock]);
    ResultCube.write(bP->my_state->avg_err);
    bP++;
  }
}
void TranslatorsFlowByFlow::TranslateResults_CubeToRegion(LayoutCubeWithRegions<float> & ResultCube, void * bkinfo, size_t flowIdxInBlock, size_t regId)
{

  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);
  size_t numLBeads = myJob.getNumBeads();
  BeadParams *bP = myJob.getBeadParams();

  ResultCube.setRWStrideZ();
  assert(ResultCube.getDimZ() == Result_NUM_PARAMS);

  for(size_t b = 0; b < numLBeads; b++){
    //ResultCube.setRWPtrRegion(regId,bP->x,bP->y,ResultAmpl);
    bP->Ampl[flowIdxInBlock] = ResultCube.getAtReg(regId,bP->x,bP->y,ResultAmpl);  // TODO: remove this plane and use AMPL
    bP->kmult[flowIdxInBlock] = ResultCube.getAtReg(regId,bP->x,bP->y,ResultKmult);
    bP->my_state->avg_err = ResultCube.getAtReg(regId,bP->x,bP->y,ResultAvgErr);
    bP++;
  }
}



void TranslatorsFlowByFlow::TranslateRegionParams_CubeToRegion(LayoutCubeWithRegions<reg_params> & RegionCube,  reg_params * rP, size_t regId)
{
  *rP = RegionCube.getAtReg(regId);
}

void TranslatorsFlowByFlow::TranslateRegionParams_RegionToCube( LayoutCubeWithRegions<reg_params> & RegionCube, void* bkinfo,
    size_t regId)
{
  WorkSet myJob((BkgModelWorkInfo*)bkinfo);
  RegionCube[regId] = *(myJob.getRegionParams());
}



void TranslatorsFlowByFlow::TranslateRegionFrameCube_RegionToCube( LayoutCubeWithRegions<float> & RegionFrameCube, void * bkinfo,
     size_t regId)
{

  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

  size_t numFrames = myJob.getNumFrames();

  //frames by region by param cuda
  RegionFrameCube.setRWStrideX();

  //DeltaFrames
  float * ptr = myJob.getDeltaFrames();
  RegionFrameCube.setRWPtrRegion(regId,0,0,RfDeltaFrames);
  RegionFrameCube.writeByStride(ptr,numFrames);

  //deltaFramesStd
  if(myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace()){
    ptr = myJob.GetStdTimeCompDeltaFrame();
    RegionFrameCube.setRWPtrRegion(regId,0,0,RfDeltaFramesStd);
    RegionFrameCube.writeByStride(ptr,numFrames);
  }


  ptr = myJob.getFrameNumber();
  RegionFrameCube.setRWPtrRegion(regId,0,0,RfFrameNumber);
  RegionFrameCube.writeByStride(ptr,numFrames);

  ptr = myJob.getDarkMatter();
  for(int vec = 0; vec < 4; vec++){
    //RegionFrameCube.setRWPtr(0,regId,RfDarkMatter0+vec);
    RegionFrameCube.setRWPtrRegion(regId,0,0,RfDarkMatter0+vec);
    RegionFrameCube.writeByStride(ptr,numFrames);
    ptr += numFrames; //shift to next PCA vector
  }
}

void TranslatorsFlowByFlow::TranslateRegionFramesPerPoint_RegionToCube( LayoutCubeWithRegions<int> & RegionFramesPerPoint, void * bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

  size_t numFrames = myJob.getNumFrames();


  int * FppStd = (myJob.performExpTailFitting() && !(myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace())) ? myJob.GetETFFramesPerPoint() : myJob.GetStdFramesPerPoint();

  //frames by region by param cuda
  RegionFramesPerPoint.setRWStrideX();
  RegionFramesPerPoint.setRWPtrRegion(regId);
  RegionFramesPerPoint.writeByStride(FppStd,numFrames);
}

void TranslatorsFlowByFlow::TranslateEmphasis_RegionToCube(LayoutCubeWithRegions<float> & RegionEmphasis, void * bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);
  float * ptr =myJob.getEmphVec();
  int numFrames = myJob.getNumFrames();

  RegionEmphasis.setRWPtrRegion(regId);
  RegionEmphasis.setRWStrideX();

  for(int f=0; f< MAX_POISSON_TABLE_COL* numFrames ; f++)
  {
    RegionEmphasis.write(ptr[f]);
  }

}

void TranslatorsFlowByFlow::TranslateNonZeroEmphasisFrames_RegionToCube(LayoutCubeWithRegions<int> & RegionNonZeroEmphFrames, void * bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

  RegionNonZeroEmphFrames.setRWStrideX();

  RegionNonZeroEmphFrames.setRWPtrRegion(regId,0,0,NzEmphFrames);
  int * ptr = myJob.GetNonZeroEmphasisFrames();
  for(int i = 0; i < MAX_POISSON_TABLE_COL; i++){
    RegionNonZeroEmphFrames.write(ptr[i]);
  }

  if (myJob.performExpTailFitting() && myJob.performRecompressionTailRawTrace()) {
    RegionNonZeroEmphFrames.setRWPtrRegion(regId,0,0,NzEmphFramesStd);
    ptr = myJob.GetNonZeroEmphasisFramesForStdCompression();
    for(int i = 0; i < MAX_POISSON_TABLE_COL; i++){
      RegionNonZeroEmphFrames.write(ptr[i]);
    }
  }
}


void TranslatorsFlowByFlow::TranslateNucRise_RegionToCube(LayoutCubeWithRegions<float> & NucRise, void *bkinfo, size_t flowIdx, size_t regId)
{

  //float * nucRise,  size_t numFrames, size_t regId)
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

  int numFrames = myJob.getNumFrames();
  float * ptr = info->bkgObj->region_data->my_regions.cache_step.nuc_rise_fine_step;
  ptr += flowIdx*numFrames*ISIG_SUB_STEPS_SINGLE_FLOW, myJob.getNumFrames();


  NucRise.setRWPtrRegion(regId);
  NucRise.setRWStrideX();

  for(int f=0; f< ISIG_SUB_STEPS_SINGLE_FLOW*numFrames; f++)
  {
    NucRise.write(ptr[f]);
  }


}


void TranslatorsFlowByFlow::TranslatePerFlowRegionParams_RegionToCube(LayoutCubeWithRegions<PerFlowParamsRegion> & PerFlowParamReg, void * bkinfo, size_t flowIdx, size_t regId )
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;

  reg_params * rp = &(info->bkgObj->region_data->my_regions.rp);

  PerFlowParamsRegion & ref = PerFlowParamReg.refAtReg(regId);

  ref.setCopyDrift(rp->CopyDrift);
  ref.setDarkness(rp->darkness[0]);
  ref.setRatioDrift(rp->RatioDrift);
  ref.setSigma(*(rp->AccessSigma()));
  ref.setStart(info->bkgObj->region_data->my_regions.cache_step.i_start_fine_step[flowIdx]);
  ref.setTMidNuc(rp->AccessTMidNuc()[0]);
  ref.setTMidNucShift(rp->nuc_shape.t_mid_nuc_shift_per_flow[flowIdx]);
  ref.setTshift(rp->tshift);

#if TRANSLATE_DEBUG_OUTPUT
  cout << "DEBUG regId " << regId << " ";
  ref.print();
#endif

}
void UpdatePerFlowRegionParams_RegionToCube(LayoutCubeWithRegions<PerFlowParamsRegion> & PerFlowParamReg, reg_params * rp, size_t flowIdx, size_t regId )
{
  //BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;

//  reg_params * rp = &(info->bkgObj->region_data->my_regions.rp);

  PerFlowParamsRegion & ref = PerFlowParamReg.refAtReg(regId);
  ref.setTMidNucShift(rp->nuc_shape.t_mid_nuc_shift_per_flow[flowIdx]);

}




void TranslatorsFlowByFlow::TranslateConstantRegionParams_RegionToCube(LayoutCubeWithRegions<ConstantParamsRegion> & ConstParamReg, void * bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;

  reg_params * rp = &(info->bkgObj->region_data->my_regions.rp);
  ConstantParamsRegion & ref = ConstParamReg.refAtReg(regId);

  ref.setMoleculesToMicromolarConversion(rp->molecules_to_micromolar_conversion);
  ref.setSens(rp->sens);
  ref.setTauE(rp->tauE);
  ref.setTauRM(rp->tau_R_m);
  ref.setTauRO(rp->tau_R_o);
  ref.setTimeStart(info->bkgObj->region_data->time_c.time_start);
  ref.setT0Frame(info->bkgObj->region_data->t0_frame);
  ref.setMinTmidNuc(info->bkgObj->region_data->my_regions.rp_low.AccessTMidNuc()[0]);
  ref.setMaxTmidNuc(info->bkgObj->region_data->my_regions.rp_high.AccessTMidNuc()[0]);
  ref.setMinCopyDrift(*(info->bkgObj->region_data->my_regions.rp_low.AccessCopyDrift()));
  ref.setMaxCopyDrift((*info->bkgObj->region_data->my_regions.rp_high.AccessCopyDrift()));
  ref.setMinRatioDrift(*(info->bkgObj->region_data->my_regions.rp_low.AccessRatioDrift()));
  ref.setMaxRatioDrift(*(info->bkgObj->region_data->my_regions.rp_high.AccessRatioDrift()));
#if TRANSLATE_DEBUG_OUTPUT
  cout << "DEBUG regId " << regId << " ";
  ref.print();
#endif

}

void TranslatorsFlowByFlow::TranslatePerNucRegionParams_RegionToCube(LayoutCubeWithRegions<PerNucParamsRegion> & PerNucCube, void * bkinfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  reg_params * rp = &(info->bkgObj->region_data->my_regions.rp);

  PerNucCube.setRWStrideZ(); //step through planes, each plane one nuc
  PerNucCube.setRWPtrRegion(regId);

  for(int i = 0; i < NUMNUC; i++ ){
    PerNucParamsRegion & ref = PerNucCube.ref();
    ref.setD(rp->AccessD()[i]);
    ref.setKmax(rp->kmax[i]);
    ref.setKrate(rp->krate[i]);
    ref.setNucModifyRatio(rp->AccessNucModifyRatio()[i]);
    ref.setTMidNucDelay(rp->nuc_shape.t_mid_nuc_delay[i]);
    ref.setC(rp->nuc_shape.C[i]);
    ref.setSigmaMult(rp->nuc_shape.sigma_mult[i]);
#if TRANSLATE_DEBUG_OUTPUT
    cout << "DEBUG regId " << regId << " NucId " << i << " ";
    ref.print();
#endif
  }
}




void TranslatorsFlowByFlow::TranslatePerFlowRegionParams_CubeToRegion(LayoutCubeWithRegions<PerFlowParamsRegion> &perFlowRegParams, void *bkgInfo, size_t regId)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkgInfo;
  WorkSet myJob(info);
  reg_params *rp = myJob.getRegionParams();

  PerFlowParamsRegion pfRegP = perFlowRegParams.getAtReg(regId);

  *(rp->AccessTMidNuc()) = pfRegP.getTMidNuc();
  *(rp->AccessRatioDrift()) = pfRegP.getRatioDrift();
  *(rp->AccessCopyDrift()) = pfRegP.getCopyDrift();
  *(rp->AccessTMidNucShiftPerFlow()) = pfRegP.getTMidNucShift();
}












void ConstanSymbolCopier::PopulateSymbolConstantImgageParams(ImgRegParams iP, ConstantFrameParams & CfP, void * bkinfoArray)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfoArray;

  int maxFrames = 0;

  for(size_t i=0; i < iP.getNumRegions(); i++)
  {
    int f = info[i].bkgObj->region_data->time_c.npts();
    maxFrames = (maxFrames <f )?(f):(maxFrames);
  }

  RawImage * rpt = info->img->raw;

  CfP.setRawFrames(rpt->frames);
  CfP.setUncompFrames(rpt->uncompFrames);
  if(CfP.getUncompFrames() > MAX_UNCOMPRESSED_FRAMES_GPU){
    cout <<"---------------------------------------------------------------------------"<<endl
         <<"CUDA WARNING: The number of uncompressed frames of "<< CfP.getUncompFrames() <<" for this block " << endl
         <<"              exceeds the GPU frame buffer limit for a maximum of " << MAX_UNCOMPRESSED_FRAMES_GPU << " frames." <<endl
         <<"              No more than "<< MAX_UNCOMPRESSED_FRAMES_GPU <<" uncompressed frames will used!!" <<endl
         <<"---------------------------------------------------------------------------"<<endl;
    CfP.setUncompFrames(MAX_UNCOMPRESSED_FRAMES_GPU);
  }

  CfP.setMaxCompFrames(maxFrames);


  for(int i=0; i < rpt->uncompFrames; i++){
    CfP.interpolatedFrames[i] =  rpt->interpolatedFrames[i];
    CfP.interpolatedMult[i] =  rpt->interpolatedMult[i];
    CfP.interpolatedDiv[i] =  rpt->interpolatedDiv[i];
  }

  CfP.print();
  copySymbolsToDevice(CfP);

}

void ConstanSymbolCopier::PopulateSymbolConstantGlobal( ConstantParamsGlobal & CpG, void * bkinfo)
{

  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);
  reg_params * rp = myJob.getRegionParams();

  CpG.setAdjKmult(myJob.getkmultAdj());
  CpG.setMinKmult(myJob.getkmultLowLimit());
  CpG.setMaxKmult(myJob.getkmultHighLimit());
  CpG.setMinAmpl(myJob.getAmpLowLimit());

  CpG.setMaxTauB(rp->max_tauB);
  CpG.setMinTauB(rp->min_tauB);

  CpG.setScaleLimit(myJob.expTailFitBkgAdjLimit());
  CpG.setTailDClowerBound(myJob.expTailFitBkgDcLowerLimit());

  CpG.setMagicDivisorForTiming(rp->nuc_shape.magic_divisor_for_timing);
  CpG.setNucFlowSpan(rp->nuc_shape.nuc_flow_span);
  CpG.setValveOpen(rp->nuc_shape.valve_open);

  CpG.setEmphWidth(myJob.getEmphasisData().emphasis_width);
  CpG.setEmphAmpl(myJob.getEmphasisData().emphasis_ampl);
  CpG.setEmphParams(myJob.getEmphasisData().emp);

  CpG.setClonalFilterFirstFlow(info->polyclonal_filter_opts.mixed_first_flow);
  CpG.setClonalFilterLastFlow(info->polyclonal_filter_opts.mixed_last_flow);

  CpG.print();

  copySymbolsToDevice(CpG);

}

void ConstanSymbolCopier::PopulateSymbolPerFlowGlobal(PerFlowParamsGlobal & pFpG, void * bkinfo)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);

  pFpG.setRealFnum(myJob.getAbsoluteFlowNum());
  pFpG.setFlowIdx(0); // ToDo remove when data only copied by flow
  //pFpG.setNucId(myJob.getFlowIdxMap()[flowIdx]);
  pFpG.setNucId(myJob.getNucIdForFlow(myJob.getAbsoluteFlowNum()));
  pFpG.print();
  copySymbolsToDevice(pFpG);

}

void ConstanSymbolCopier::PopulateSymbolConfigParams(ConfigParams & confP, void * bkinfo)
{
  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);
  reg_params * rp = myJob.getRegionParams();

  confP.clear();
  if(myJob.fitkmultAlways()) confP.setFitKmult();
  if(rp->fit_taue) confP.setFitTauE();
  if(myJob.performExpTailFitting()) confP.setPerformExpTailFitting();
  if(myJob.performBkgAdjInExpTailFit()) confP.setPerformBkgAdjInExpTailFit();
  if(rp->use_alternative_etbR_equation) confP.setUseAlternativeEtbRequation();
  if(myJob.useDarkMatterPCA()) confP.setUseDarkMatterPCA();
  if(myJob.useDynamicEmphasis()) confP.setUseDynamicEmphasis();
  if(myJob.performRecompressionTailRawTrace()) confP.setPerformRecompressTailRawTrace();
  if(myJob.performCrossTalkCorrection()) confP.setPerformTraceLevelXTalk();
  else if(myJob.performWellsLevelXTalkCorrection()) confP.setPerformWellsLevelXTalk();  //ToDo: Wells Level is default and will not be set if Trace level is already set
  if(myJob.performPolyClonalFilter()) confP.setPerformPolyClonalFilter();

  confP.print();
  copySymbolsToDevice(confP);
}





/*void ConstanSymbolCopier::PopulateSymbolConstantRegParamBounds( ConstantRegParamBounds & CpB, void * bkinfo)
{

  BkgModelWorkInfo* info = (BkgModelWorkInfo*)bkinfo;
  WorkSet myJob(info);
  reg_params * rpMin = myJob.getRegionParamMinBounds();
  reg_params * rpMax = myJob.getRegionParamMaxBounds();

  CpB.setMinTmidNuc(rpMin->AccessTMidNuc()[0]);
  CpB.setMaxTmidNuc(rpMax->AccessTMidNuc()[0]);
  CpB.setMinRatioDrift(rpMin->AccessRatioDrift()[0]);
  CpB.setMinCopyDrift(rpMin->AccessCopyDrift()[0]);
  CpB.setMaxCopyDrift(rpMax->AccessCopyDrift()[0]);

  CpB.print();

  copySymbolsToDevice(CpB);

}*/




void BuildGenericSampleMask(
  bool * sampleMask, //global base pointer to mask Initialized with false
  const ImgRegParams &imgP,
  size_t regId)
{

   for (int sampIdx = 0; sampIdx < 100; sampIdx ++){
     size_t large_num = sampIdx*104729;
     size_t sampRow = large_num / imgP.getRegH(regId);
     sampRow = sampRow % imgP.getRegH(regId);
     size_t sampCol = large_num % imgP.getRegW(regId);
//     cout << "sample id " << sampIdx << ": " << imgP.getWellIdx(regId,sampCol,sampRow) << endl;
     sampleMask[ imgP.getWellIdx(regId,sampCol,sampRow) ] = true;
//     cout << "sample id " << sampIdx << " mask update done" << endl;
   }
}






void BuildMaskFromBeadParams_RegionToCube(   LayoutCubeWithRegions<unsigned short> & Mask,
    size_t numLBeads,
    BeadParams * bP,
    size_t regId)
{
  size_t x,y;
//  ImgRegParams ImageParams = Mask.getParams();

  for(size_t idx = 0; idx < numLBeads; idx++  ){
    x = bP->x;
    y = bP->y;
    Mask.setRWPtrRegion(regId,x,y);
    Mask.write(MaskLive);
    bP++;
  }
}

//////////////////////////////////////////////////////////////////////
//CUBE PER FLOW DUMPER CLASS
//this class can be used to dump multiple flows in the new layout cube design from the old bkinfo object in a random order.

template <typename T>
CubePerFlowDump<T>::CubePerFlowDump(size_t planeW, size_t planeH, size_t regW, size_t regH, size_t planes, size_t numFlowsinBlock):
flowBlockBase(0),regionsDumped(0),regionsDumpedLastBlock(0),FlowBlockSize(numFlowsinBlock),filePathPrefix("FlowPlaneDump")
{ // change from plane to cube
  ImageParams.init(planeW,planeH,regW,regH);
  for(size_t i=0; i< FlowBlockSize; i++)
    FlowCubes.push_back (new LayoutCubeWithRegions<T>(planeW, planeH, regW, regH,planes,HostMem));
};

template <typename T>
CubePerFlowDump<T>::CubePerFlowDump( ImgRegParams iP, size_t planes, size_t numFlowsinBlock):
flowBlockBase(0),regionsDumped(0),regionsDumpedLastBlock(0),FlowBlockSize(numFlowsinBlock),filePathPrefix("FlowPlaneDump")
{ // change from plane to cube
  ImageParams = iP;
  for(size_t i=0; i< FlowBlockSize; i++)
    FlowCubes.push_back (new LayoutCubeWithRegions<T>(iP.getImgW(), iP.getImgH(), iP.getRegW(),iP.getRegH(),planes,HostMem));
};


template <typename T>
CubePerFlowDump<T>::~CubePerFlowDump(){
  destroy();
}

template <typename T>
void CubePerFlowDump<T>::destroy(){
  while(FlowCubes.size() > 0){
    delete *FlowCubes.begin();
    FlowCubes.erase(FlowCubes.begin());
  }
}

template <typename T>
void  CubePerFlowDump<T>::setFilePathPrefix(string filep)
{
  filePathPrefix = filep;
}

template <typename T>
void CubePerFlowDump<T>::WriteOneFlowToFile(LayoutCubeWithRegions<T> * dumpCube, size_t dumpflow)
{
  ostringstream filename;
  filename << DUMP_PATH << "/" << filePathPrefix  << dumpflow << ".dat";
  ofstream myFile (filename.str().c_str(), ios::binary);
  cout << filename.str() << ": writing flow cube for " << regionsDumped << " regions at flow " << dumpflow << endl;
  dumpCube->dumpCubeToFile(myFile);
  myFile.close();
}


template <typename T>
void CubePerFlowDump<T>::WriteAllFlowsToFile(){
  for(size_t f = 0; f<FlowBlockSize; f++){
    WriteOneFlowToFile( FlowCubes[f],  flowBlockBase+f);
  }
}

template <typename T>
void CubePerFlowDump<T>::ClearFlowCubes(){
  for(size_t f = 0; f<FlowBlockSize; f++){
    FlowCubes[f]->memSet(0);
  }
}

template <typename T>
void CubePerFlowDump<T>::DumpFlowBlockRegion(size_t regId, T* data, size_t flowBlockBegin, size_t nPerFlow, size_t flowstride,  size_t plane )
{
  if(nPerFlow > flowstride) flowstride = nPerFlow;

  if(flowBlockBegin != flowBlockBase){
    if (regionsDumped != regionsDumpedLastBlock ){
      regionsDumpedLastBlock = regionsDumped;
      WriteAllFlowsToFile();
    }
    flowBlockBase = flowBlockBegin;
    regionsDumped = 0;
  }

  //size_t regId = ImageParams.getRegId(regCol,regRow);
  //ToDo get rid of constparam structure and extract only needed params
  for(size_t f = 0; f<FlowBlockSize; f++){
    FlowCubes[f]->setRWStrideX();
    FlowCubes[f]->setRWPtrRegion(regId,0,0,plane);
    for(size_t w=0; w < nPerFlow; w++)
      FlowCubes[f]->write(data[w]);
    data += flowstride;
  }
  regionsDumped++;

  if(regionsDumped == ImageParams.getNumRegions() || regionsDumped == regionsDumpedLastBlock){
    regionsDumpedLastBlock = regionsDumped; //set here to prevent writing same data again in next flow block
    WriteAllFlowsToFile();
  }
}

template <typename T>
void CubePerFlowDump<T>::DumpOneFlowRegion(size_t regId, LayoutCubeWithRegions<T> & input, size_t iRegId, size_t flowBlockBegin, size_t flowInBlockIdx,  size_t startPlane,  size_t numPlanes)
{

  size_t f = flowInBlockIdx;
  assert(FlowCubes[f]->getRegW(regId) == input.getRegW(iRegId) && FlowCubes[f]->getRegH(regId) == input.getRegH(iRegId)); //check for identical region size
  assert( startPlane+numPlanes <= FlowCubes[f]->getDimZ()); //check for enough planes
  assert( startPlane+numPlanes <= input.getDimZ()); //check for enough planes

  if(flowBlockBegin != flowBlockBase){
    if (regionsDumped != regionsDumpedLastBlock ){
      regionsDumpedLastBlock = regionsDumped;
      WriteAllFlowsToFile();
    }
    flowBlockBase = flowBlockBegin;
    regionsDumped = 0;
  }

  FlowCubes[f]->copyReg(regId,input,iRegId,numPlanes,startPlane);

  if(f== FlowBlockSize-1) regionsDumped++; //hacky and requires one region dumps all flowblocksize flows sequentially

  if(regionsDumped > 0){
    if(regionsDumped == ImageParams.getNumRegions() || regionsDumped == regionsDumpedLastBlock){
      regionsDumpedLastBlock = regionsDumped; //set here to prevent writing same data again in next flow block
      WriteAllFlowsToFile();
    }
  }
}

template <typename T>
void CubePerFlowDump<T>::DumpOneFlowBlock(LayoutCubeWithRegions<T> & input, size_t flowBlockStartFlow, size_t flowInBlockIdx)
{

  flowBlockBase = flowBlockStartFlow;
  FlowCubes[flowInBlockIdx]->copy(input);
  regionsDumped = input.getParams().getNumRegions();
  WriteOneFlowToFile(FlowCubes[flowInBlockIdx], flowBlockStartFlow + flowInBlockIdx);

}


template <typename T>
void CubePerFlowDump<T>::DumpFlowBlockRegion(size_t ax, size_t ay, T* data, size_t realflowIdx, size_t nPerFlow, size_t flowstride )
{
  assert(ax < ImageParams.getImgW() && ay < ImageParams.getImgH());
  size_t regId = ImageParams.getRegId(ax,ay);
  DumpFlowBlockRegion(regId, data, realflowIdx,nPerFlow, flowstride );
}

template <typename T>
void CubePerFlowDump<T>::ReadInOneFlow(size_t realflowIdx)
{

  ostringstream filename;
  filename << DUMP_PATH  << "/" << filePathPrefix << realflowIdx << ".dat";

  ifstream myFile (filename.str().c_str(), ios::binary);

  if(!myFile){
    cerr << "file " << filename.str() << " could not be opened!" << endl;
    exit (-1);
  }

  cout << "reading data at flow " << realflowIdx << " from file " << filename.str() << endl;


  if(!FlowCubes[0]->readCubeFromFile(myFile))
  {
    cout << "Error reading flow " << realflowIdx << " from file " << filename.str() << " buffer dimensions missmatch!" << endl;
    myFile.close();
    exit(-1);
  }

  myFile.close();
}

template <typename T>
LayoutCubeWithRegions<T> & CubePerFlowDump<T>::getFlowCube(size_t realflowIdx)
{
  if(realflowIdx != flowBlockBase){
    ReadInOneFlow(realflowIdx);

  }
  flowBlockBase = realflowIdx;
  return *(FlowCubes[0]);
}


///////////////////////////////////////
//Explicit declaration
/*
template class LayoutCube<short>;
template class LayoutCube<unsigned short>;
template class LayoutCube<int>;
template class LayoutCube<size_t>;
template class LayoutCube<float>;
template class LayoutCube<ConstantParamsRegion>;
template class LayoutCube<PerFlowParamsRegion>;
template class LayoutCube<PerNucParamsRegion>;
template class LayoutCube<SampleCoordPair>;
*/
template class CubePerFlowDump<float>;
template class CubePerFlowDump<short>;
template class CubePerFlowDump<reg_params>;








