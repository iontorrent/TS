/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ImageSpecClass.h"


ImageSpecClass::ImageSpecClass()
{
  timestamps = NULL;
  rows = cols = 0;
  scale_of_chip =0;
  uncompFrames = 0;
  vfr_enabled = true;
  acqPrefix = strdup("acq_");
}

int ImageSpecClass::LeadTimeForChipSize()
{
  int totalMem = totalMemOnTorrentServer();
  if (totalMem > (24*1024*1024)) {
    if ((rows * cols) > 10000000)
      return(10); // 318
    if ((rows * cols) > 2000000)
      return(8); // 316
    else
      return(20);
  }
  else {
    if ((rows * cols) > 10000000)
      return(2); // 318
    if ((rows * cols) > 2000000)
      return(4); // 316
    else
      return(20);
  }
}

/********************************************************************
             // Open an image file to get some parameters for the dataset
             // use the first nuke flow file instead of beadfind file.
 ********************************************************************/
void ImageSpecClass::DeriveSpecsFromDat(CommandLineOpts &clo, int numFlows, char *experimentName)
{
  Image img;
  img.SetImgLoadImmediate(false);
  img.SetNumAcqFiles(numFlows);
  img.SetIgnoreChecksumErrors(clo.img_control.ignoreChecksumErrors);
  char *firstAcqFile = (char *) malloc(strlen(clo.sys_context.dat_source_directory) + strlen(
                                         acqPrefix) + 10);
  sprintf(firstAcqFile, "%s/%s%04d.dat", clo.sys_context.dat_source_directory, acqPrefix, 0);

  if (!img.LoadRaw(firstAcqFile, 0, true, false))
  {
    exit(EXIT_FAILURE);
  }
  img.SetOffsetFromChipOrigin(firstAcqFile);
  free(firstAcqFile);

  img.SetDir(experimentName);
  img.SetFlowOffset(clo.img_control.flowTimeOffset);

  // grab rows & cols here - as good a place as any
  rows = img.GetImage()->rows;
  cols = img.GetImage()->cols;
  scale_of_chip = rows*cols;
  clo.loc_context.chip_offset_x = img.GetImage()->chip_offset_x;
  clo.loc_context.chip_offset_y = img.GetImage()->chip_offset_y;
  clo.loc_context.rows = rows;
  clo.loc_context.cols = cols;

  vfr_enabled= img.VFREnabled();

  // this globally limits the Loadraw method and other methods to process only this many frames (for speed)
  clo.img_control.totalFrames = img.GetMaxFrames();
  if (clo.img_control.maxFrames != 0)
  {
    // command-line override of the frames to analyze
    clo.img_control.maxFrames = (clo.img_control.maxFrames > clo.img_control.totalFrames ? clo.img_control.totalFrames
                     : clo.img_control.maxFrames);
    img.SetMaxFrames(clo.img_control.maxFrames);
  }
  else
  {
    clo.img_control.maxFrames = clo.img_control.totalFrames; // set to total frames in image.
  }
  uncompFrames = img.GetUnCompFrames();
  timestamps = new int[clo.img_control.maxFrames];
  memcpy(timestamps,img.GetImage()->timestamps,sizeof(int)*clo.img_control.maxFrames);
  // Deallocate image memory
  img.Close();
}

ImageSpecClass::~ImageSpecClass()
{
  if (timestamps!=NULL) delete[] timestamps;
  if (acqPrefix !=NULL) free(acqPrefix);
}

