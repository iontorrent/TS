/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "BeadfindControlOpts.h"
#include "Utils.h"
void BeadfindControlOpts::DefaultBeadfindControl()
{
  //maxNumKeyFlows = 0;
  //minNumKeyFlows = 99;
  bfMinLiveRatio = .0001;
  bfMinLiveLibSnr = 4;
  bfMinLiveTfSnr = 4;
  bfTfFilterQuantile = 1;
  bfLibFilterQuantile = 1;
  skipBeadfindSdRecover = 0;
  beadfindThumbnail = 0;
  beadfindLagOneFilt = 0;
  beadMaskFile = NULL;
  maskFileCategorized = 0;
  sprintf (bfFileBase, "beadfind_post_0003.dat");
  sprintf (preRunbfFileBase, "beadfind_pre_0003.dat");
  BF_ADVANCED = true;
  SINGLEBF = true;
  noduds = 0;
  bfOutputDebug = false;
  if (isInternalServer()) {
    bfOutputDebug = true;
  }
  beadfindType = "differential";
  useSignalReference = true;
}

BeadfindControlOpts::~BeadfindControlOpts()
{
  if (beadMaskFile)
    free (beadMaskFile);
}
