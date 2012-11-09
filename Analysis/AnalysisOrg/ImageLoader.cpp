/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ImageLoader.h"
#include "Utils.h"
#include "SynchDatSerialize.h"

void ImageTracker::WaitForFlowToLoad ( int flow ) // absolute flow value
{
   int flow_buffer_for_flow = FlowBufferFromFlow(flow);  // temporary while we resolve the confusion of buffers and flows
 // disable compiler optimizing this loop away
  while ( ! ( ( int volatile * ) CurRead ) [flow_buffer_for_flow] ) {
    sleep ( 1 ); // wait for the load worker to load the current image
  }
}

void ImageTracker::FireUpThreads()
{
  if (doingSdat)
  {
    cout <<  "Doing sdats" << endl;
    pthread_create (&loaderThread, NULL, FileSDatLoader, &master_img_loader);
  }
  else
  {
    cout <<  "Doing regular dats" << endl;
    pthread_create (&loaderThread, NULL, FileLoader, &master_img_loader);
  }
}

void ImageTracker::AllocateImageBuffers(int ignoreChecksumErrors, int total_timeout)
{
  img = new Image[flow_buffer_size];
  for ( int n = 0; n < flow_buffer_size; n++ ) {
    img[n].SetImgLoadImmediate ( false );
    img[n].SetNumAcqFiles ( flow_buffer_size );
    img[n].SetIgnoreChecksumErrors ( ignoreChecksumErrors );
    if (total_timeout > 0)
      img[n].SetTimeout(img[n].GetRetryInterval(), total_timeout);
  }
}

void ImageTracker::AllocateReadAndProcessFlags()
{
    CurRead = new unsigned int [flow_buffer_size];
  CurProcessed = new unsigned int [flow_buffer_size];

  memset ( CurRead, 0, flow_buffer_size*sizeof ( unsigned int ) );
  memset ( CurProcessed, 0, flow_buffer_size*sizeof ( unsigned int ) );
}

void ImageTracker::AllocateSdatBuffers()
{
    sdat = new SynchDat[flow_buffer_size];
}

void ImageTracker::NothingInit()
{
  img = NULL;
  sdat = NULL;
  CurRead = NULL;
  CurProcessed=NULL;
  doingSdat = false;
}

ImageTracker::ImageTracker ( int _flow_buffer_size, int ignoreChecksumErrors, bool _doingSdat, int total_timeout )
{
  flow_buffer_size = _flow_buffer_size;

  NothingInit();
  doingSdat = _doingSdat;
  
  AllocateImageBuffers(ignoreChecksumErrors, total_timeout);
  
  AllocateReadAndProcessFlags();

  AllocateSdatBuffers();

}

int ImageTracker::FlowBufferFromFlow(int flow)
{
  return(flow-master_img_loader.startingFlow);
}

void ImageTracker::FinishFlow ( int flow )
{
  int flow_buffer_for_flow = FlowBufferFromFlow(flow);
  
  img[flow_buffer_for_flow].Close();
  sdat[flow_buffer_for_flow].Close();
   ( ( int volatile * ) CurProcessed ) [flow_buffer_for_flow] = 1;
}

void ImageTracker::DeleteFlags()
{
    if ( CurRead !=NULL ) delete[] CurRead;
  if ( CurProcessed!=NULL ) delete[] CurProcessed;
  CurRead = NULL;
  CurProcessed = NULL;
}

void ImageTracker::DeleteImageBuffers()
{

  if ( img!=NULL ) delete[] img;
  img=NULL;
}

void ImageTracker::DeleteSdatBuffers()
{
  if ( sdat != NULL ) delete [] sdat;
  sdat = NULL;
}

ImageTracker::~ImageTracker()
{
  // spin down our threads when we go out of scope
  pthread_join (loaderThread, NULL);

  DeleteImageBuffers();
  
  DeleteFlags();
  DeleteSdatBuffers();
}



void ImageTracker::SetUpImageLoaderInfo (CommandLineOpts &inception_state,
                            ComplexMask &a_complex_mask, 
                            ImageSpecClass &my_image_spec)
{

  int normStart = 5;
  int normEnd = 20;
  int standard_smooth_span = 15;


  master_img_loader.type = imageLoadAllE;
  master_img_loader.flow = -1;  // nonsense currently, used in individual loading
  master_img_loader.cur_buffer = -1; //nonsense currently, used in individual loading
  master_img_loader.flow_buffer_size = flow_buffer_size; // not the same!!!!!!!
  master_img_loader.startingFlow = inception_state.flow_context.startingFlow;
  
  master_img_loader.img = img;  // just use ImageTracker object instead?
  master_img_loader.sdat = sdat;
  master_img_loader.doingSdat = doingSdat;
  
  master_img_loader.pinnedInFlow = a_complex_mask.pinnedInFlow;
  master_img_loader.mask = a_complex_mask.my_mask;

  master_img_loader.normStart = normStart;
  master_img_loader.normEnd = normEnd;
  master_img_loader.NNinnerx = inception_state.img_control.NNinnerx;
  master_img_loader.NNinnery = inception_state.img_control.NNinnery;
  master_img_loader.NNouterx = inception_state.img_control.NNouterx;
  master_img_loader.NNoutery = inception_state.img_control.NNoutery;
  master_img_loader.smooth_span = standard_smooth_span;
  
  master_img_loader.CurRead = & ( CurRead[0] );
  master_img_loader.CurProcessed = & ( CurProcessed[0] );
  
  master_img_loader.dat_source_directory = inception_state.sys_context.dat_source_directory;
  master_img_loader.acqPrefix = inception_state.img_control.acqPrefix;
  
  master_img_loader.numFlowsPerCycle = inception_state.flow_context.numFlowsPerCycle; // @TODO: really?  is this even used correctly
  master_img_loader.hasWashFlow = inception_state.img_control.has_wash_flow;  
  
  master_img_loader.finished = false;
  master_img_loader.lead = ( inception_state.bkg_control.readaheadDat != 0 ) ? inception_state.bkg_control.readaheadDat : my_image_spec.LeadTimeForChipSize();
  master_img_loader.inception_state = &inception_state;  // why must we pass globals around everywhere?
  
  printf ( "Subtract Empties: %d\n", inception_state.img_control.nn_subtract_empties );
  master_img_loader.doRawBkgSubtract = ( inception_state.img_control.nn_subtract_empties>0 );
  master_img_loader.doEmptyWellNormalization = inception_state.bkg_control.empty_well_normalization;
}

void ImageTracker::DecideOnRawDatsToBufferForThisFlowBlock()
{
  // need dynamic read ahead every 20 block of flows for 318 chips
  if (ChipIdDecoder::GetGlobalChipId() == ChipId318) {
    static int readahead = master_img_loader.lead;
    const double allowedFreeRatio = 0.2;
    double freeSystemMemory = GetAbsoluteFreeSystemMemoryInKB() / (1024.0*1024.0);
    double totalSystemMemory = (double)(totalMemOnTorrentServer()) / (1024.0*1024.0);

    double freeRatio = freeSystemMemory / totalSystemMemory; 

    if (freeRatio < allowedFreeRatio)
      master_img_loader.lead = 1;
    else if (freeRatio < 0.3)
      master_img_loader.lead = 2;
    else if (freeRatio < 0.6)
      master_img_loader.lead = 4;
    else 
      master_img_loader.lead = readahead;
    printf("TotalMem: %f FreeMem: %f Free/total Ratio: %f Allowed Free/total Ratio: %f readahead: %d\n", 
        totalSystemMemory, freeSystemMemory, freeRatio, allowedFreeRatio, master_img_loader.lead);
    fflush(stdout);
  }
}
