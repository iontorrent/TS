/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#include "ImageLoaderQueue.h"
#include "Utils.h"
#include "FlowBuffer.h"
#include "ComparatorNoiseCorrector.h"
#include "CorrNoiseCorrector.h"
#include "PairPixelXtalkCorrector.h"
#include "AdvCompr.h"
#include "FlowSequence.h"
#include "FluidPotentialCorrector.h"
#include <sys/fcntl.h>
#include <sys/prctl.h>
#include "crop/Acq.h"
#include "ChipIdDecoder.h"

typedef struct {
  int threadNum;
  WorkerInfoQueue *wq;
}WqInfo_t;

void JustLoadOneImageWithPinnedUpdate(ImageLoadWorkInfo *cur_image_loader);

// handle our fake mutex flags
void SetReadCompleted(ImageLoadWorkInfo *one_img_loader)
{
  // just a note: no guarantee that prior flows have finished yet
  ( ( int volatile * ) one_img_loader->CurRead ) [one_img_loader->cur_buffer] = 1;
}


void *FileLoadWorker ( void *arg )
{
  WqInfo_t *wqinfo = (WqInfo_t *)arg;
  int threadNum=wqinfo->threadNum;
  WorkerInfoQueue *q = wqinfo->wq;
  assert ( q );

  bool done = false;
  char dateStr[256];
  struct tm newtime;
  time_t ltime;
  double T1=0,T2=0,T3,T4;

  char name[20];
  sprintf(name,"FileLdWkr%d",threadNum);
  prctl(PR_SET_NAME,name,0,0,0);

  const double noiseThreshold = 0.;  // to be set by command line option
  FluidPotentialCorrector fpCorr(noiseThreshold);

  while ( !done )
  {
    WorkerInfoQueueItem item = q->GetItem();

    if ( item.finished == true )
    {
      // we are no longer needed...go away!
      done = true;
      q->DecrementDone();
      continue;
    }
    ImageLoadWorkInfo *one_img_loader = ( ImageLoadWorkInfo * ) item.private_data;

    ClockTimer timer;
    Timer tmr;
    Image *img = &one_img_loader->img[one_img_loader->cur_buffer];

    if (one_img_loader->inception_state->img_control.threaded_file_access)
    {
      tmr.restart();
      if ( !img->LoadRaw ( one_img_loader->name) )
      {
        fprintf ( stdout, "ERROR: Failed to load raw image %s\n", one_img_loader->name );
        exit ( EXIT_FAILURE );
      }

      T1=tmr.elapsed();
      tmr.restart();
      if(img->raw->imageState & IMAGESTATE_QuickPinnedPixelDetect)
        one_img_loader->pinnedInFlow->QuickUpdate ( one_img_loader->flow, img);
      else
        one_img_loader->pinnedInFlow->Update ( one_img_loader->flow, img,(ImageTransformer::gain_correction?ImageTransformer::gain_correction:0));

      T2=tmr.elapsed();
    }

    tmr.restart();


    // correct in-channel electrical cross-talk
    ImageTransformer::XTChannelCorrect ( one_img_loader->img[one_img_loader->cur_buffer].raw, one_img_loader->img[one_img_loader->cur_buffer].results_folder ); // buffer_ix

    // testing of lossy compression
    if(ImageTransformer::PCATest[0]) {
      AdvComprTest(one_img_loader->name,&one_img_loader->img[one_img_loader->cur_buffer],ImageTransformer::PCATest,false/*one_img_loader->inception_state->img_control.col_flicker_correct*/ );
    }
    // col noise correction (if done during lossy compression will already have happened.
    else if ( !(img->raw->imageState & IMAGESTATE_ComparatorCorrected) &&
              one_img_loader->inception_state->img_control.col_flicker_correct )
    {
        if( one_img_loader->inception_state->img_control.col_pair_pixel_xtalk_correct ){
            PairPixelXtalkCorrector xtalkCorrector;
            float xtalk_fraction = one_img_loader->inception_state->img_control.pair_xtalk_fraction;
            xtalkCorrector.Correct(one_img_loader->img[one_img_loader->cur_buffer].raw, xtalk_fraction);
        }
        if (one_img_loader->inception_state->img_control.corr_noise_correct){
      	  CorrNoiseCorrector rnc;
      	  rnc.CorrectCorrNoise(one_img_loader->img[one_img_loader->cur_buffer].raw,3,one_img_loader->inception_state->bfd_control.beadfindThumbnail );
        }


      if(one_img_loader->inception_state->bfd_control.beadfindThumbnail)
      {
        //ComparatorNoiseCorrector
        ComparatorNoiseCorrector cnc;
        cnc.CorrectComparatorNoiseThumbnail(one_img_loader->img[one_img_loader->cur_buffer].raw, one_img_loader->mask, one_img_loader->inception_state->loc_context.regionXSize,one_img_loader->inception_state->loc_context.regionYSize, one_img_loader->inception_state->img_control.col_flicker_correct_verbose);
      } else {
        ComparatorNoiseCorrector cnc;
        cnc.CorrectComparatorNoise(one_img_loader->img[one_img_loader->cur_buffer].raw, one_img_loader->mask, one_img_loader->inception_state->img_control.col_flicker_correct_verbose, one_img_loader->inception_state->img_control.aggressive_cnc,false,threadNum );
      }
    }
//#define DEBUG_IMAGE_CORR_ISSUES 1
#ifdef DEBUG_IMAGE_CORR_ISSUES

        char newName[1024];
        char *nptr,*ptr=one_img_loader->name;
        while((nptr = strstr(ptr,"/")))ptr=nptr+1;
        if(*ptr == '/')
        	ptr++;
        sprintf(newName,"%s_proc",one_img_loader->name);

        Acq saver;
        saver.SetData ( &one_img_loader->img[one_img_loader->cur_buffer] );
        saver.WriteVFC(newName, 0, 0, one_img_loader->img[one_img_loader->cur_buffer].raw->cols, one_img_loader->img[one_img_loader->cur_buffer].raw->rows);
#endif
    T3=tmr.elapsed();
    tmr.restart();

    // Fluid potential corrector
    const bool correctFluidPotential = one_img_loader->inception_state->img_control.fluid_potential_correct;
    const double noiseThreshold = (double) one_img_loader->inception_state->img_control.fluid_potential_threshold;
    if (correctFluidPotential){
        fpCorr.setThreshold(noiseThreshold);

                // parse rowsum file name to load
                const std::string datFileName = one_img_loader->name;
                printf("dat file name %s\n", datFileName.c_str());
                const size_t pos1 = datFileName.find_last_of('/');
                const size_t pos2 = datFileName.find_last_of('.');
                const std::string rowsumFileName = datFileName.substr(0, pos1) + "/../rowsum/" + datFileName.substr(pos1+1, pos2-pos1-1) + ".hwsum";
                // printf("rowsum file name %s pos1 %u pos2 %u\n", rowsumFileName.c_str(), (uint32_t) pos1, (uint32_t) pos2);


                // determine if the data file is thumbnail or block
        const bool isThumbnail = one_img_loader->inception_state->bfd_control.beadfindThumbnail;
        if(isThumbnail){
                fpCorr.setIsThumbnail();
        }

        // set image
        const unsigned int regionXSize = one_img_loader->inception_state->loc_context.regionXSize;
        const unsigned int regionYSize = one_img_loader->inception_state->loc_context.regionYSize;
        const char nucChar = one_img_loader->inception_state->flow_context.ReturnNucForNthFlow(one_img_loader->flow);
        fpCorr.setImage(one_img_loader->img[one_img_loader->cur_buffer].raw, one_img_loader->mask,  regionYSize, regionXSize, nucChar);
        //printf("nucChar is %c\n", nucChar);


        // load sensing electrode data
        if (isThumbnail){
                const unsigned int numRows = one_img_loader->img[one_img_loader->cur_buffer].raw->rows;
                fpCorr.loadSensingElectrodeDataThumbnail(rowsumFileName, numRows);

        } else {
                // determine startRow and endRow
                one_img_loader->img[one_img_loader->cur_buffer].SetOffsetFromChipOrigin(one_img_loader->name);
                const unsigned int startRow = one_img_loader->img[one_img_loader->cur_buffer].raw->chip_offset_y;
                const unsigned int endRow = startRow + one_img_loader->img[one_img_loader->cur_buffer].raw->rows;
       //         printf("offset x y = %d %d, start and end rows are %u,%u\n", one_img_loader->img[one_img_loader->cur_buffer].raw->chip_offset_x,one_img_loader->img[one_img_loader->cur_buffer].raw->chip_offset_y, startRow, endRow);
                fpCorr.loadSensingElectrodeData(rowsumFileName, startRow, endRow);
        }

        // correct fluid potential
         if (fpCorr.readRowSumIsSuccess()){
                 fpCorr.doCorrection();
         } else {
                 printf("fluidPotentialCorrector skipped:  Cannot find rowsum file %s \n", rowsumFileName.c_str());
         }

    }


    // dump dc offset one_img_loaderrmation before we do any normalization
    DumpDcOffset ( one_img_loader );
    int buffer_ix = one_img_loader->cur_buffer;

    // setting the mean of frames to zero will be done by the bkgmodel as soon as its loaded.

// the traces will be zero'd by the bknd model loader anyway, no need to do them here.
 //   img->SetMeanOfFramesToZero ( one_img_loader->normStart, one_img_loader->normEnd,0 );


    // calculate the smooth pH step amplitude in empty wells across the whole image
    if ( one_img_loader->doEmptyWellNormalization )
    {
      // this is redundant with what's in the EmptyTrace class, but this method of spatial normalization
      // is going to go away soon, so not worth fixing
      MaskType referenceMask = MaskReference;

      if ( one_img_loader->inception_state->bkg_control.trace_control.use_dud_and_empty_wells_as_reference )
        referenceMask = ( MaskType ) ( MaskReference | MaskDud );

      one_img_loader->img[buffer_ix].CalculateEmptyWellLocalScaleForFlow ( * ( one_img_loader->pinnedInFlow ),one_img_loader->mask,one_img_loader->flow,referenceMask,one_img_loader->smooth_span );
#if 0
      char fname[512];
      sprintf ( fname,"ewampimg_%04d.txt",one_img_loader->flow );
      FILE *ewampfile = fopen ( fname,"w" );
      for ( int row=0;row < one_img_loader->img[buffer_ix].GetRows();row++ )
      {
        int col;

        for ( col=0;col < ( one_img_loader->img[buffer_ix].GetCols()-1 ); col++ )
          fprintf ( ewampfile,"%9.5f\t",one_img_loader->img[buffer_ix].getEmptyWellAmplitude ( row,col ) );

        fprintf ( ewampfile,"%9.5f\n",one_img_loader->img[buffer_ix].getEmptyWellAmplitude ( row,col ) );
      }
      fclose ( ewampfile );
#endif
    }

    // dump pH step debug one_img_loader to file (its really fast, don't worry)
    DumpStep ( one_img_loader );
    if ( one_img_loader->doRawBkgSubtract )
    {
      one_img_loader->img[buffer_ix].SubtractLocalReferenceTrace ( one_img_loader->mask, MaskBead, MaskReference, one_img_loader->NNinnerx,
                                                                   one_img_loader->NNinnery,one_img_loader->NNouterx,one_img_loader->NNoutery, false, true, true );
      if ( one_img_loader->flow==0 ) // absolute first flow,not flow buffer
        printf ( "Notify user: NN empty subtraction in effect.  No further warnings will be given. %d \n",one_img_loader->flow );
    }
    T4=tmr.elapsed();

    //    printf ( "Allow model to go %d \n", one_img_loader->flow );

    SetReadCompleted(one_img_loader);

    size_t usec = timer.GetMicroSec();

    ltime=time(&ltime);
    localtime_r(&ltime, &newtime);
    strftime(dateStr,sizeof(dateStr),"%H:%M:%S", &newtime);

    fprintf ( stdout, "FileLoadWorker: ImageProcessing time for flow %d: %0.2lf(ld=%.2f pin=%.2f cnc=%.2f xt=%.2f sem=%.2lf cache=%.2lf) sec %s\n",
              one_img_loader->flow , usec / 1.0e6, T1, T2, T3, T4, img->SemaphoreWaitTime, img->CacheAccessTime, dateStr);
    fflush(stdout);
    fprintf(stdout, "File: %s\n", one_img_loader->name);
    fflush(stdout);
    q->DecrementDone();
  }

  return ( NULL );
}

void ConstructNameForCurrentInfo(ImageLoadWorkInfo *cur_image_loader)
{
  int cur_flow = cur_image_loader->flow;
  int filenum = cur_flow + ( cur_flow / cur_image_loader->numFlowsPerCycle ) * cur_image_loader->hasWashFlow;
  sprintf (cur_image_loader->name, "%s/%s%04d.%s", cur_image_loader->dat_source_directory,cur_image_loader->acqPrefix, filenum , cur_image_loader->datPostfix);
}

void SetUpIndividualImageLoaders ( ImageLoadWorkInfo *my_img_loaders, ImageLoadWorkInfo *master_img_loader )
{
  for ( int  i_buffer = 0; i_buffer < master_img_loader->flow_buffer_size; i_buffer++ )
  {
    my_img_loaders[ i_buffer ] = *master_img_loader;

    // names of files not set here yet
    my_img_loaders[i_buffer].flow = master_img_loader->startingFlow+i_buffer; //this buffer entry points to this absolute flow
    my_img_loaders[i_buffer].flow_block_sequence = master_img_loader->flow_block_sequence;
    my_img_loaders[i_buffer].cur_buffer = i_buffer; // the actual buffer
    ConstructNameForCurrentInfo(&my_img_loaders[i_buffer]);
  }
}


void KillQueue ( WorkerInfoQueue *loadWorkQ, int numWorkers )
{
  WorkerInfoQueueItem item;
  for ( int cworker = 0; cworker < numWorkers; cworker++ )
  {
    item.private_data = NULL;
    item.finished = true;
    loadWorkQ->PutItem ( item );
  }
  // wait for all workers to exit
  loadWorkQ->WaitTillDone();
}

// fake mutex using the >individual< image loaders
void DontReadAheadOfSignalProcessing (  ImageLoadWorkInfo *info, int lead)
{
  // don't read ahead too far of BkgModel regional fitter threads
  while ( info->cur_buffer > lead &&
          ! ( ( int volatile * ) info->CurProcessed ) [info->cur_buffer - lead] )
    usleep ( 100 );
}

// don't step on the compute intensity that happens every chunk of flows
void PauseForLongCompute ( int cur_flow,  ImageLoadWorkInfo *info )
{
  // If we're not the end of the flow block, don't wait.
  if ( cur_flow + 1 != info->flow_block_sequence.BlockAtFlow( cur_flow )->end() ) return;

  // wait for the the BkgModel regional fitter threads to finish this block
  while ( ! ( (( int volatile * ) info->CurProcessed)[info->cur_buffer] ) ) 
    usleep ( 100 );
}

void JustCacheOneImage(ImageLoadWorkInfo *cur_image_loader)
{

  // make sure and take the semaphore

  // just read the file in...  It will be stored in cache
  int fd = open(cur_image_loader->name,O_RDONLY);
  if(fd >= 0)
  {
    char buf[1024*1024];
    int len;
    int totalLen=0;
    while((len = read(fd,&buf[0],sizeof(buf))) > 0)
      totalLen += len;
    printf("read %d bytes from %s\n",totalLen,cur_image_loader->name);
    close(fd);
  }
  else
  {
    printf("failed to open %s\n",cur_image_loader->name);
  }
}



void JustLoadOneImageWithPinnedUpdate(ImageLoadWorkInfo *cur_image_loader)
{

  if ( !cur_image_loader->img[cur_image_loader->cur_buffer].LoadRaw ( cur_image_loader->name) )
  {
    exit ( EXIT_FAILURE );
  }
  //tikSMoother is a no-op if there was no tikSmoothFile entered on command line
  // cur_image_loader->img[cur_image_loader->cur_buffer].SmoothMeTikhonov ( NULL,false,cur_image_loader->name);
  // if gain correction has been calculated, apply it
  // @TODO: is this correctly done before pinning status is calculated, or after like XTCorrect?
  //    if ( ImageTransformer::gain_correction != NULL )
  //      ImageTransformer::GainCorrectImage ( cur_image_loader->img[cur_image_loader->cur_buffer].raw );

  // pinning updates only need to be complete to remove indeterminacy
  // in values dumped in DumpStep in FileLoadWorker
  cur_image_loader->pinnedInFlow->Update ( cur_image_loader->flow, &cur_image_loader->img[cur_image_loader->cur_buffer],ImageTransformer::gain_correction);
}

void *FileLoader ( void *arg )
{
  ImageLoadWorkInfo *master_img_loader = ( ImageLoadWorkInfo * ) arg;

  prctl(PR_SET_NAME,"FileLoader",0,0,0);


  WorkerInfoQueue *loadWorkQ = new WorkerInfoQueue ( master_img_loader->flow_buffer_size );


  ImageLoadWorkInfo *n_image_loaders = new ImageLoadWorkInfo[master_img_loader->flow_buffer_size];
  SetUpIndividualImageLoaders ( n_image_loaders,master_img_loader );


  //   int numWorkers = 1;
  int numWorkers = numCores() /4; // @TODO - this should be subject to inception_state options
  numWorkers = ( numWorkers < 4 ? 4:numWorkers );
  fprintf ( stdout, "FileLoader: numWorkers threads = %d\n", numWorkers );
  WqInfo_t wq_info[numWorkers];

  {
    int cworker;
    pthread_t work_thread;
    // spawn threads for doing background correction/fitting work
    for ( cworker = 0; cworker < numWorkers; cworker++ )
    {
      wq_info[cworker].threadNum=cworker;
      wq_info[cworker].wq = loadWorkQ;
      int t = pthread_create ( &work_thread, NULL, FileLoadWorker,
                               &wq_info[cworker] );
      pthread_detach(work_thread);
      if ( t )
        fprintf ( stderr, "Error starting thread\n" );
    }
  }

  WorkerInfoQueueItem item;

  //time_t start, end;
  int flow_buffer_size = master_img_loader->flow_buffer_size;

  // this loop goes over the individual image loaders
  for ( int i_buffer = 0; i_buffer < flow_buffer_size; i_buffer++ )
  {
    ImageLoadWorkInfo *cur_image_loader = &n_image_loaders[i_buffer];

    int cur_flow = cur_image_loader->flow;  // each job is an n_image_loaders item

    DontReadAheadOfSignalProcessing (cur_image_loader, master_img_loader->lead);
    //***We are doing this on this thread so we >load< in sequential order that pinned in Flow updates in sequential order
    if (!cur_image_loader->inception_state->img_control.threaded_file_access) {
      JustLoadOneImageWithPinnedUpdate(cur_image_loader);
    }
    //*** now we can do the rest of the computation for an image, including dumping in a multiply threaded fashion

    item.finished = false;
    item.private_data = cur_image_loader;
    loadWorkQ->PutItem ( item );

    if (!ChipIdDecoder::IsProtonChip())
      PauseForLongCompute ( cur_flow,cur_image_loader );
  }

  // wait for all of the images to be processed
  loadWorkQ->WaitTillDone();
  KillQueue ( loadWorkQ,numWorkers );

  delete loadWorkQ;
  delete[] n_image_loaders;

  master_img_loader->finished = true;

  return NULL;
}


//@TODO: why is this not a method of info?
void DumpStep ( ImageLoadWorkInfo *info )
{
  //@TODO: if only there were some object, say a flow context, that had
  // already converted the flowOrder into the string of nucleotides to
  // the length of all the flows we wouldn't need to write code like
  // this...but that would be crazy talk.
  char nucChar = info->inception_state->flow_context.ReturnNucForNthFlow ( info->flow );
  string nucStepDir = string ( info->inception_state->sys_context.GetResultsFolder() ) + string ( "/NucStep" );

  // Make output directory or quit
  if ( mkdir ( nucStepDir.c_str(), 0777 ) && ( errno != EEXIST ) )
  {
    perror ( nucStepDir.c_str() );
    return;
  }

  // Set region width & height
  int rWidth=50;
  int rHeight=50;

  // Lower left corner of the region should equal 0 modulus xModulus or yModulus
  int xModulus=50;
  int yModulus=50;

  vector<string> regionName;
  vector<int> regionStartCol;
  vector<int> regionStartRow;
  vector<int> regionWidth;
  vector<int> regionHeight;


  const RawImage *rawImg = info->img[info->cur_buffer].GetImage();

  // region choice based on chip type
  //@TODO: why is this checked for every image rather than storing the same regions for access in all flows?

  if ( ChipIdDecoder::IsProtonChip())

  {
    // Proton chips
    regionName.push_back ( "inlet" );
    regionStartCol.push_back ( xModulus * floor ( 0.1 * ( float ) rawImg->cols / ( float ) xModulus ) );
    regionStartRow.push_back ( yModulus * floor ( 0.5 * ( float ) rawImg->rows / ( float ) yModulus ) );
    regionWidth.push_back ( rWidth );
    regionHeight.push_back ( rHeight );

    regionName.push_back ( "middle" );
    regionStartCol.push_back ( xModulus * floor ( 0.5 * ( float ) rawImg->cols / ( float ) xModulus ) );
    regionStartRow.push_back ( yModulus * floor ( 0.5 * ( float ) rawImg->rows / ( float ) yModulus ) );
    regionWidth.push_back ( rWidth );
    regionHeight.push_back ( rHeight );

    regionName.push_back ( "outlet" );
    regionStartCol.push_back ( xModulus * floor ( 0.9 * ( float ) rawImg->cols / ( float ) xModulus ) );
    regionStartRow.push_back ( yModulus * floor ( 0.5 * ( float ) rawImg->rows / ( float ) yModulus ) );
    regionWidth.push_back ( rWidth );
    regionHeight.push_back ( rHeight );
  }
  else
  {
    // PGM chips
    regionName.push_back ( "inlet" );
    regionStartCol.push_back ( xModulus * floor ( 0.9 * ( float ) rawImg->cols / ( float ) xModulus ) );
    regionStartRow.push_back ( yModulus * floor ( 0.9 * ( float ) rawImg->rows / ( float ) yModulus ) );
    regionWidth.push_back ( rWidth );
    regionHeight.push_back ( rHeight );

    regionName.push_back ( "middle" );
    regionStartCol.push_back ( xModulus * floor ( 0.5 * ( float ) rawImg->cols / ( float ) xModulus ) );
    regionStartRow.push_back ( yModulus * floor ( 0.5 * ( float ) rawImg->rows / ( float ) yModulus ) );
    regionWidth.push_back ( rWidth );
    regionHeight.push_back ( rHeight );

    regionName.push_back ( "outlet" );
    regionStartCol.push_back ( xModulus * floor ( 0.1 * ( float ) rawImg->cols / ( float ) xModulus ) );
    regionStartRow.push_back ( yModulus * floor ( 0.1 * ( float ) rawImg->rows / ( float ) yModulus ) );
    regionWidth.push_back ( rWidth );
    regionHeight.push_back ( rHeight );
  }

  for ( unsigned int iRegion=0; iRegion<regionStartRow.size(); iRegion++ )
  {
    info->img[info->cur_buffer].DumpStep (
          regionStartCol[iRegion],
          regionStartRow[iRegion],
          regionWidth[iRegion],
          regionHeight[iRegion],
          regionName[iRegion],
          nucChar,
          nucStepDir,
          info->mask,
          info->pinnedInFlow,
          info->flow
          );
  }
}

void DumpDcOffset ( ImageLoadWorkInfo *info )
{
  char nucChar = info->inception_state->flow_context.ReturnNucForNthFlow ( info->flow );
  string dcOffsetDir = string ( info->inception_state->sys_context.GetResultsFolder() ) + string ( "/dcOffset" );

  // Make output directory or quit
  if ( mkdir ( dcOffsetDir.c_str(), 0777 ) && ( errno != EEXIST ) )
  {
    perror ( dcOffsetDir.c_str() );
    return;
  }

  int nSample=100000;

  info->img[info->cur_buffer].DumpDcOffset (
        nSample,
        dcOffsetDir,
        nucChar,
        info->flow
        );
}

