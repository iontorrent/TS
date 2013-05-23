/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "TrackProgress.h"

TrackProgress::TrackProgress(){
  fpLog = NULL;
  time(&analysis_start_time);
  time(&analysis_current_time);
}

TrackProgress::~TrackProgress(){
  fprintf(stdout, "Completion Time = %s\n", ctime(&analysis_current_time));
  fflush (stdout);
  fclose(fpLog);
}

void TrackProgress::ReportState(char *my_state){
      time(&analysis_current_time);
      fprintf(stdout, "\n%s: Elapsed: %.1lf minutes\n\n", my_state, difftime(analysis_current_time, analysis_start_time) / 60);
      fprintf(fpLog, "%s = %.1lf minutes\n", my_state, difftime(analysis_current_time, analysis_start_time) / 60);
      fflush(NULL);
}

void TrackProgress::InitFPLog (CommandLineOpts &inception_state)
{
  char file[] = "processParameters.txt";
  char *fileName = ( char * ) malloc ( strlen ( inception_state.sys_context.results_folder ) + strlen ( file ) + 2 );
  sprintf ( fileName, "%s/%s", inception_state.sys_context.results_folder, file );
  fopen_s ( &fpLog, fileName, "a" );
  if ( !fpLog ) {
    perror ( fileName );
    exit ( errno );
  }
  free ( fileName );
  fileName = NULL;
}

void TrackProgress::WriteProcessParameters (CommandLineOpts &inception_state)
{
  //  Dump the processing parameters to a file
  fprintf ( fpLog, "[global]\n" );
  fprintf ( fpLog, "Command line = %s\n", inception_state.GetCmdLine().c_str() );
  fprintf ( fpLog, "dataDirectory = %s\n", inception_state.sys_context.dat_source_directory );
  fprintf ( fpLog, "Smoothing File = %s\n", inception_state.img_control.tikSmoothingFile );
  fprintf ( fpLog, "runId = %s\n", inception_state.sys_context.runId );
  fprintf ( fpLog, "flowOrder = %s\n", inception_state.flow_context.flowOrder );
  fprintf ( fpLog, "washFlow = %d\n", inception_state.img_control.has_wash_flow );
  fprintf ( fpLog, "libraryKey = %s\n", inception_state.key_context.libKey );
  fprintf ( fpLog, "tfKey = %s\n", inception_state.key_context.tfKey );
  fprintf ( fpLog, "minNumKeyFlows = %d\n", inception_state.key_context.minNumKeyFlows );
  fprintf ( fpLog, "maxNumKeyFlows = %d\n", inception_state.key_context.maxNumKeyFlows );
  fprintf ( fpLog, "nokey = %s\n", (inception_state.bkg_control.nokey ? "true":"false" ) );
  fprintf ( fpLog, "numFlows = %d\n", inception_state.flow_context.numTotalFlows );
  fprintf ( fpLog, "cyclesProcessed = %d\n", inception_state.flow_context.numTotalFlows/4 ); // @TODO: may conflict with PGM now
  fprintf ( fpLog, "framesProcessed = %d\n", inception_state.img_control.maxFrames );
  fprintf ( fpLog, "framesInData = %d\n", inception_state.img_control.totalFrames );

  fprintf ( fpLog, "bkgModelUsed = %s\n", "yes" );

  fprintf ( fpLog, "nucTraceCorrectionUsed = %s\n", ( inception_state.no_control.NUC_TRACE_CORRECT ? "true":"false" ) );

  fprintf ( fpLog, "nearest-neighborParameters = Inner: (%d,%d) Outer: (%d,%d)\n", inception_state.img_control.NNinnerx, inception_state.img_control.NNinnery, inception_state.img_control.NNouterx, inception_state.img_control.NNoutery );

  fprintf ( fpLog, "Advanced beadfind = %s\n", inception_state.bfd_control.BF_ADVANCED ? "enabled":"disabled" );
  fprintf ( fpLog, "use pinned wells = %s\n", inception_state.no_control.USE_PINNED ? "true":"false" );
  fprintf ( fpLog, "use exclusion mask = %s\n", inception_state.loc_context.exclusionMaskSet ? "true":"false" );
  fprintf ( fpLog, "Version = %s\n", IonVersion::GetVersion().c_str() );
  fprintf ( fpLog, "Build = %s\n", IonVersion::GetBuildNum().c_str() );
  fprintf ( fpLog, "SvnRev = %s\n", IonVersion::GetSvnRev().c_str() );

  fprintf ( fpLog, "Chip = %d,%d\n", inception_state.loc_context.chip_len_x,inception_state.loc_context.chip_len_y );
  fprintf ( fpLog, "Block = %d,%d,%d,%d\n", inception_state.loc_context.chip_offset_x, inception_state.loc_context.chip_offset_y, inception_state.loc_context.cols, inception_state.loc_context.rows );
  for ( int q=0;q<inception_state.loc_context.numCropRegions;q++ )
    fprintf ( fpLog, "Cropped Region = %d,%d,%d,%d\n", inception_state.loc_context.cropRegions[q].col, inception_state.loc_context.cropRegions[q].row, inception_state.loc_context.cropRegions[q].w, inception_state.loc_context.cropRegions[q].h );

  fprintf ( fpLog, "Analysis Region = %d,%d,%d,%d\n", inception_state.loc_context.chipRegion.col, inception_state.loc_context.chipRegion.row, inception_state.loc_context.chipRegion.col+inception_state.loc_context.chipRegion.w, inception_state.loc_context.chipRegion.row+inception_state.loc_context.chipRegion.h );
  fprintf ( fpLog, "numRegions = %d\n", inception_state.loc_context.numRegions );
  fprintf ( fpLog, "regionRows = %d\nregionCols = %d\n", inception_state.loc_context.regionsY, inception_state.loc_context.regionsX );
  fprintf ( fpLog, "regionSize = %dx%d\n", inception_state.loc_context.regionXSize, inception_state.loc_context.regionYSize );
  //fprintf (fpLog, "\tRow Column Height Width\n");
  //for (int i=0;i<numRegions;i++)
  //  fprintf (fpLog, "[%3d] %5d %5d %5d %5d\n", i, regions[i].row, regions[i].col,regions[i].h,regions[i].w);
  fflush ( NULL );
}

