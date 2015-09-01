/*
 * enumDefines.h
 *
 *  Created on: Jun 4, 2014
 *      Author: jakob
 */

#ifndef ENUMDEFINES_H_
#define ENUMDEFINES_H_



// BkgModelMask
enum BkgModelMaskType {
  BkgMaskNone                = 0,
  BkgMaskBadRead               = ( 1<<0 ), // set in BFMask instead
  BkgMaskPolyClonal            = ( 1<<1 ),
  BkgMaskCorrupt               = ( 1<<2 ),   //update BFMaks washout instead
  BkgMaskRandomSample          = ( 1<<3 ),
  BkgMaskHighQaulity           = ( 1<<4 ),
  BkgMaskRegionalSampled       = ( 1<<5 ),
  BkgMaskPinned                = ( 1<<7 ),
  BkgMaskAll                   = 0xffff
};

//ToDo: determine validity of thresholds!
#define THRESHOLD_T0_AVERAGE 2.0f
#define THRESHOLD_NUM_EMPTIES 5
#define THRESHOLD_NUM_REGION_SAMPLE 5

// BkgModelMask
enum RegionStateMask {
  RegionMaskLive                            = 0,
  RegionMaskNoLiveBeads                     = ( 1<<0 ),
  RegionMaskNoT0Average                     = ( 1<<1 ),
  RegionMaskT0AverageBelowThreshold         = ( 1<<2 ),
  RegionMaskNoEmpties                       = ( 1<<3 ),
  RegionMaskNumEmptiesBelowThreshold        = ( 1<<4 ),
  RegionMaskNoRegionSamples                 = ( 1<<5 ),
  RegionMaskNumRegionSamplesBelowThreshold  = ( 1<<6 ),
  RegionMaskAll                             = 0xffff
};



//////////////////////////////////////////////////
//parameter Cube Plane Ids

//cube with dimensions: width: maxNumFrames, height: numRegions, depth: Rf_NUM_PARAMS
enum RegionFramePlanes{
  RfDeltaFrames = 0,
  RfDeltaFramesStd ,
  RfFrameNumber ,
  RfDarkMatter0,
  RfDarkMatter1 ,
  RfDarkMatter2 ,
  RfDarkMatter3 ,
  Rf_NUM_PARAMS
};
//RfFramesPerPoint, //either std or etf frames per point

//cube with dimensions: width: imageWidth, height: ImageHeight, depth: bp_NUM_PARAMS
enum BeadParamPlanes{
  BpCopies = 0,
  BpR ,
  BpDmult ,
  BpGain ,
  BpTauAdj ,
  BpPhi ,
  BpPCAval0 ,
  BpPCAval1 ,
  BpPCAval2,
  BpPCAval3,
  Bp_NUM_PARAMS
};

//cube with dimensions: width: imageWidth, height: ImageHeight, depth: Bs_NUM_PARAMS
enum PolyClonalPlanes{
  PolyPpf = 0, // updated on per flow basis
  PolySsq , // updated on per flow basis
  PolyKeyNorm, // key norm determined during first 20 flows
  Poly_NUM_PARAMS
};

enum NonZeroEmphPlanes{
  NzEmphFrames = 0,
  NzEmphFramesStd ,
  Nz_NUM_PARAMS
};


//cube with dimensions: width: imageWidth, height: ImageHeight, depth: Result_NUM_PARAMS
enum ResultPlanes{
  ResultAmpl = 0,
  ResultKmult ,
  ResultAvgErr,
  ResultAmplCopyMapXTalk,
  ResultAmplXTalk,
  Result_NUM_PARAMS
};




//cube per nuc params plane Ids
enum NucParams{
  NucD = 0,    // dNTP diffusion rate
  NucKmax,  // saturation per nuc action rate
  NucKrate,  // rate of incorporation
  NucT_mid_nuc_delay,
  NucModifyRatio, // buffering modifier per nuc
  Nuc_NUM_PARAMS
};


#endif /* ENUMDEFINES_H_ */
