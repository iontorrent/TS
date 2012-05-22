/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <deque>
#include "BkgFitterTracker.h"
#include "BkgModel.h"
#include "BkgModelHdf5.h"
#include "EmptyTraceTracker.h"
#include "GlobalDefaultsFromBkgControl.h"
#include "Image.h"
#include "IonErr.h"
#include "json/json.h"
#include "MaskFunctions.h"
#include "mixed.h"
#include "SeparatorInterface.h"
#include "TrackProgress.h"
#include "WellFileManipulation.h"

using namespace std;

void ApplyClonalFilter(Mask& mask, const char* experimentName, BkgModel *BkgModelFitters[], Region* regions, int numRegions, bool doClonalFilter, int flow);
void ApplyClonalFilter(Mask& mask, BkgModel *BkgModelFitters[], Region* regions, int numRegions, const deque<float>& ppf, const deque<float>& ssq);
void UpdateMask(Mask& mask, BkgModel *BkgModelFitters[], Region* regions, int numRegions);
void GetFilterTrainingSample(deque<int>& row, deque<int>& col, deque<float>& ppf, deque<float>& ssq, deque<float>& nrm, BkgModel *BkgModelFitters[], Region* regions, int numRegions);
void DumpPPFSSQ (const char* experimentName, const deque<int>& row, const deque<int>& col, const deque<float>& ppf, const deque<float>& ssq, const deque<float>& nrm);

//@TODO: bad coding practice to have side effects like this does in the static parameters of a model
// this is just a way of disguising global variables
void SetUpToProcessImages (ImageSpecClass &my_image_spec, CommandLineOpts &clo, char *experimentName, TrackProgress &my_progress)
{
  // set up to process images aka 'dat' files.
  ExportSubRegionSpecsToImage (clo.loc_context);

  // make sure we're using XTCorrection at the right offset for cropped regions
  Image::SetCroppedRegionOrigin (clo.loc_context.cropped_region_x_offset,clo.loc_context.cropped_region_y_offset);
  Image::CalibrateChannelXTCorrection (clo.sys_context.dat_source_directory,"lsrowimage.dat");

  //@TODO: this mess has nasty side effects on the arguments.
  my_image_spec.DeriveSpecsFromDat (clo.sys_context, clo.img_control, clo.loc_context,1,experimentName);   // dummy - only reads 1 dat file

  fprintf (my_progress.fpLog, "VFR = %s\n", my_image_spec.vfr_enabled ? "enabled":"disabled");    // always enabled these days, useless?
}

void CheckReplay(CommandLineOpts& clo)
{
  assert ( (clo.bkg_control.replayBkgModelData
	    & clo.bkg_control.recordBkgModelData) != true );
  if (clo.bkg_control.recordBkgModelData){
    H5ReplayRecorder rr = H5ReplayRecorder(clo);
    rr.CreateFile(); // create file to record to
  }
 if (clo.bkg_control.replayBkgModelData){
   H5ReplayReader rr  = H5ReplayReader(clo);  // checks file to replay from exists
 }
}


void DoThreadedBackgroundModel (RawWells &rawWells, CommandLineOpts &clo, Mask *maskPtr, char *experimentName, int numFlows, char *chipType,
                                ImageSpecClass &my_image_spec, std::vector<float> &smooth_t0_est, Region *regions, int totalRegions, RegionTiming *region_timing,SeqListClass &my_keys,std::vector<float> *tauB=NULL, std::vector<float> *tauE=NULL)
{

  MemUsage ("StartingBackground");


  int my_nuc_block[NUMFB];
  // TODO: Objects should be isolated!!!!
  GlobalDefaultsForBkgModel::SetFlowOrder (clo.flow_context.flowOrder); // @TODO: 2nd duplicated code instance
  GlobalDefaultsForBkgModel::GetFlowOrderBlock (my_nuc_block,0,NUMFB);
  // @TODO these matrices never get updated for later blocks of nucs and should be updated during iterations
  InitializeLevMarSparseMatrices (my_nuc_block);

  SetBkgModelGlobalDefaults (clo.bkg_control,chipType,experimentName);

  BkgFitterTracker GlobalFitter (totalRegions);
  // plan
  GlobalFitter.PlanComputation (clo.bkg_control);
  // Setup the output flows for background parameters. @todoThis should be wrapped up in an object                              BgkParamH5 bgParamH5;
  MemUsage ("BeforeBgInitialization");

  BkgParamH5 bgParamH5;
  string hgBgDbgFile = "";
  if (clo.bkg_control.bkgModelHdf5Debug)
  {
    hgBgDbgFile = bgParamH5.Init (clo.sys_context,clo.loc_context, numFlows, my_image_spec);
  }

  CheckReplay(clo);

  ImageTracker my_img_set (numFlows,clo.img_control.ignoreChecksumErrors, maskPtr);
    
  EmptyTraceTracker my_emptytrace_trk(regions, region_timing, totalRegions, smooth_t0_est, my_image_spec, clo);

  GlobalFitter.ThreadedInitialization (rawWells, clo, maskPtr, my_img_set.pinnedInFlow, experimentName, my_image_spec,smooth_t0_est,regions,totalRegions, region_timing, my_keys,&bgParamH5.ptrs,  my_emptytrace_trk, tauB,tauE);

  MemUsage ("AfterBgInitialization");
  // Image Loading thread setup to grab flows in the background

  ImageLoadWorkInfo glinfo;
  SetUpImageLoaderInfo (glinfo,clo, maskPtr, my_img_set, my_image_spec, my_emptytrace_trk, numFlows);

  pthread_t loaderThread;
  pthread_create (&loaderThread, NULL, FileLoader, &glinfo);

  // Now do threaded solving, going through all the flows

  GlobalFitter.SpinUp();
  // need to have initialized the regions for this
  GlobalFitter.SetRegionProcessOrder ();

  // process all flows...
  for (int flow = 0; flow < numFlows; flow++)
  {
    time_t flow_start;
    time_t flow_end;
    time (&flow_start);

    // coordinate with the ImageLoader threads for this flow to be read in
    // WaitForFlowToLoad guarantees all flows up this one have been read in
    my_img_set.WaitForFlowToLoad (flow);
    bool last = ( (flow) == (numFlows - 1));

    GlobalFitter.ExecuteFitForFlow (flow,my_img_set,last); // isolate this object so it can carry out actions in any order it chooses.

    MemUsage ("Memory_Flow: " + ToStr (flow));
    time (&flow_end);
    fprintf (stdout, "ProcessImage compute time for flow %d: %0.3lf sec.\n",
             flow, difftime (flow_end, flow_start));


    // capture the regional parameters every 20 flows, plus one bead per region at "random"
    // @TODO replace with clean hdf5 interface for sampling beads and region parameters
    GlobalFitter.DumpBkgModelRegionInfo (experimentName,flow,last);
    GlobalFitter.DumpBkgModelBeadInfo (experimentName,flow,last, clo.bkg_control.debug_bead_only>0);

    //@TODO:  no CLO use here - use GlobalDefaultsForBKGMODEL instead
    // Never give up just because filter failed.
    try{
        ApplyClonalFilter (*maskPtr, experimentName, GlobalFitter.BkgModelFitters, regions, totalRegions, clo.bkg_control.enableBkgModelClonalFilter, flow);
    }catch(exception& e){
        cerr << "NOTE: clonal filter failed."
             << e.what()
             << endl;
    }catch(...){
        cerr << "NOTE: clonal filter failed." << endl;
    }

    // coordinate with the ImageLoader threads that this flow is done with
    // and release resources associated with this image
    my_img_set.FinishFlow (flow);

    IncrementalWriteWells (rawWells,flow,false,clo.bkg_control.saveWellsFrequency,NUMFB,numFlows);
    
    if (clo.bkg_control.bkgModelHdf5Debug)
      bgParamH5.IncrementalWrite (GlobalFitter, flow, numFlows, last);
    // individual regions output a '.' when they finish...terminate them all with a \n to keep the
    // display clean
    // printf ("\n");  // they don't seem to do that any more
  }
  rawWells.WriteRanks();
  rawWells.WriteInfo();
  rawWells.Close();
  
  //@TODO: why is this not taken care of within the bgParamsH5 object?
  if (clo.bkg_control.bkgModelHdf5Debug)
  {
    bgParamH5.Close();
    cout << "bgParamH5 output: " << hgBgDbgFile << endl;
  }

  GlobalFitter.UnSpinGpuThreads ();

  CleanupLevMarSparseMatrices();
  GlobalDefaultsForBkgModel::StaticCleanup();
  pthread_join (loaderThread, NULL);

  if (clo.bkg_control.updateMaskAfterBkgModel)
    my_img_set.pinnedInFlow->UpdateMaskWithPinned (maskPtr); //update maskPtr

  my_img_set.pinnedInFlow->DumpSummaryPinsPerFlow (experimentName);
}


void ApplyClonalFilter (Mask& mask, const char* experimentName, BkgModel *BkgModelFitters[], Region* regions, int numRegions, bool doClonalFilter, int flow)
{
  int applyFlow = ceil(1.0*mixed_last_flow() / NUMFB) * NUMFB - 1;
  if (flow == applyFlow and doClonalFilter)
  {
    deque<int>   row;
    deque<int>   col;
    deque<float> ppf;
    deque<float> ssq;
    deque<float> nrm;
    GetFilterTrainingSample (row, col, ppf, ssq, nrm, BkgModelFitters, regions, numRegions);
    DumpPPFSSQ(experimentName, row, col, ppf, ssq, nrm);
    ApplyClonalFilter (mask, BkgModelFitters, regions, numRegions, ppf, ssq);
    UpdateMask(mask, BkgModelFitters, regions, numRegions);
  }
}

void UpdateMask(Mask& mask, BkgModel *BkgModelFitters[], Region* regions, int numRegions)
{
  for (int rgn=0; rgn<numRegions; ++rgn)
  {
    BkgModel& model     = *BkgModelFitters[rgn];
    int       numWells  = model.GetNumLiveBeads();
    int       rowOffset = regions[rgn].row;
    int       colOffset = regions[rgn].col;

    for (int well=0; well<numWells; ++well)
    {
      bead_params& bead  = model.GetParams (well);
      bead_state&  state = bead.my_state;

      // Record clonal reads in mask:
      int row = rowOffset + bead.y;
      int col = colOffset + bead.x;
      if(mask.Match(col, row, MaskLib)){
        if(state.bad_read)
          mask.Set(col, row, MaskFilteredBadKey);
        else if(state.ppf >= mixed_ppf_cutoff())
          mask.Set(col, row, MaskFilteredBadResidual);
        else if(not state.clonal_read)
          mask.Set(col, row, MaskFilteredBadPPF);
      }
    }
  }
}

void ApplyClonalFilter (Mask& mask, BkgModel *BkgModelFitters[], Region* regions, int numRegions, const deque<float>& ppf, const deque<float>& ssq)
{
  clonal_filter filter;
  filter_counts counts;
  make_filter (filter, counts, ppf, ssq);

  for (int rgn=0; rgn<numRegions; ++rgn)
  {
    BkgModel& model     = *BkgModelFitters[rgn];
    int       numWells  = model.GetNumLiveBeads();
    int       rowOffset = regions[rgn].row;
    int       colOffset = regions[rgn].col;

    for (int well=0; well<numWells; ++well)
    {
      bead_params& bead  = model.GetParams (well);
      bead_state&  state = bead.my_state;
 
      int row = rowOffset + bead.y;
      int col = colOffset + bead.x;
      if(mask.Match(col, row, MaskLib))
        state.clonal_read = filter.is_clonal (state.ppf, state.ssq);
      else if(mask.Match(col, row, MaskTF))
        state.clonal_read = true;
    }
  }
}

void GetFilterTrainingSample (deque<int>& row, deque<int>& col, deque<float>& ppf, deque<float>& ssq, deque<float>& nrm, BkgModel *BkgModelFitters[], Region* regions, int numRegions)
{
  for (int r=0; r<numRegions; ++r)
  {
    int rowOffset = regions[r].row;
    int colOffset = regions[r].col;
    int numWells  = BkgModelFitters[r]->GetNumLiveBeads();
    for (int well=0; well<numWells; ++well)
    {
      bead_params bead;
      BkgModelFitters[r]->GetParams (well, &bead);
      const bead_state& state = bead.my_state;
      if (state.random_samp and state.ppf<mixed_ppf_cutoff() and not state.bad_read)
      {
        row.push_back (rowOffset + bead.y);
        col.push_back (colOffset + bead.x);
        ppf.push_back (state.ppf);
        ssq.push_back (state.ssq);
        nrm.push_back (state.key_norm);
      }
    }
  }
}

void DumpPPFSSQ (const char* experimentName, const deque<int>& row, const deque<int>& col, const deque<float>& ppf, const deque<float>& ssq, const deque<float>& nrm)
{
  string fname = string (experimentName) + "/BkgModelFilterData.txt";
  ofstream out (fname.c_str());
  assert (out);

  deque<int>::const_iterator   r = row.begin();
  deque<int>::const_iterator   c = col.begin();
  deque<float>::const_iterator p = ppf.begin();
  deque<float>::const_iterator s = ssq.begin();
  deque<float>::const_iterator n = nrm.begin();
  for (; p!=ppf.end(); ++r, ++c, ++p, ++s, ++n)
  {
    out << setw (6) << *r
        << setw (6) << *c
        << setw (8) << setprecision (2) << fixed << *p
        << setw (8) << setprecision (2) << fixed << *s
        << setw (8) << setprecision (2) << fixed << *n
        << endl;
  }
}

// output from this are a functioning wells file and a beadfind mask
// images are only known internally to this.
void RealImagesToWells (RawWells &rawWells, Mask *maskPtr,
                        CommandLineOpts &clo,
                        char *experimentName, string &analysisLocation,
                        int numFlows,
                        SeqListClass &my_keys,
                        TrackProgress &my_progress, Region &wholeChip,
                        int &well_rows, int &well_cols)
{

  char *chipType = GetChipId (clo.sys_context.dat_source_directory);
  ChipIdDecoder::SetGlobalChipId (chipType);

  ImageSpecClass my_image_spec;
  SetUpToProcessImages (my_image_spec, clo, experimentName, my_progress);

  SetUpRegionsForAnalysis (my_image_spec.rows,my_image_spec.cols, clo.loc_context, wholeChip);

  SetExcludeMask (clo,maskPtr,chipType,my_image_spec.rows,my_image_spec.cols);

  // Write processParameters.parse file now that processing is about to begin
  clo.WriteProcessParameters();

  // region definitions, in theory shared between background model and beadfind
  int totalRegions = clo.loc_context.numRegions;  // we may omit some regions due to exclusion or other ideas
  Region region_list[totalRegions];
  SetUpRegions (region_list,my_image_spec.rows,my_image_spec.cols,clo.loc_context.regionXSize,clo.loc_context.regionYSize);
  // need crude nuc rise timing information for bkg model currently
  RegionTiming region_timing[totalRegions];
  // global timing estimates - need to have to pass to bkgmodel
  std::vector<float> smooth_t0_est;
  // possibly need these in bkg model
  std::vector<float> tauB, tauE;


  /*********************************************************************
  // Beadfind Section
   *********************************************************************/

  if (clo.bfd_control.beadfindType == "differential")
  {
    DifferentialSeparator *diffSeparator=NULL;

    diffSeparator = new DifferentialSeparator();
    DoDiffSeparatorFromCLO (diffSeparator, clo, maskPtr, analysisLocation, my_keys.seqList,my_keys.numSeqListItems);
    // now actually set up the mask I want
    maskPtr->Copy (diffSeparator->GetMask());
    my_progress.ReportState ("Beadfind Complete");


    SetupForBkgModelTiming (diffSeparator, smooth_t0_est, region_timing,
                            region_list, totalRegions, my_image_spec, maskPtr, clo.bfd_control.beadfindThumbnail == 0);
    // Get the average of tauB and tauE for wells
    // Add getTausFromSeparator here

    if (clo.mod_control.passTau)
      getTausFromSeparator (maskPtr,diffSeparator,tauB,tauE);

    if (diffSeparator !=NULL)
    {
      delete diffSeparator;
      diffSeparator = NULL;
    }
    // Update progress bar status file: well proc complete/image proc started
    updateProgress (WELL_TO_IMAGE);
  }
  else
  {
    fprintf (stderr, "Don't recognize --beadfind-type %s\n", clo.bfd_control.beadfindType.c_str());
    exit (EXIT_FAILURE);
  }

  UpdateBeadFindOutcomes (maskPtr, wholeChip, experimentName, !clo.bfd_control.SINGLEBF, 0);
  my_progress.ReportState ("Bead Categorization Complete");

  if (clo.mod_control.BEADFIND_ONLY)
  {
    // Remove temporary wells file
    clo.sys_context.CleanupTmpWellsFile (false);

    fprintf (stdout,
             "Beadfind Only Mode has completed successfully\n");
    exit (EXIT_SUCCESS);
  }


  /********************************************************************
   *
   *  Background Modelling Process
   *
   *******************************************************************/

  CreateWellsFileForWriting (rawWells, maskPtr, clo, NUMFB,
                             numFlows, my_image_spec.rows, my_image_spec.cols,
                             chipType);

  // we might use some other model here
  if (clo.mod_control.USE_BKGMODEL)
  {
    // Here's the point where we really do the background model and not just setup for it
    if (clo.mod_control.passTau)
      DoThreadedBackgroundModel (rawWells, clo, maskPtr, experimentName, numFlows, chipType, my_image_spec, smooth_t0_est, region_list,totalRegions, region_timing,my_keys, &tauB, &tauE);
    else
      DoThreadedBackgroundModel (rawWells, clo, maskPtr, experimentName, numFlows, chipType, my_image_spec, smooth_t0_est, region_list,totalRegions, region_timing,my_keys, NULL, NULL);

  } // end BKG_MODL

  // whatever model we run, copy the signals to permanent location
  clo.sys_context.CopyTmpWellFileToPermanent (clo.mod_control.USE_RAWWELLS, experimentName);
  // mask file may be updated by model processing
  UpdateBeadFindOutcomes (maskPtr, wholeChip, experimentName, !clo.bfd_control.SINGLEBF, 0);


  my_progress.ReportState ("Raw flowgrams complete");
  // we deal only with wells from now on
  well_rows=my_image_spec.rows;
  well_cols=my_image_spec.cols;

  free (chipType);
}


// output from this are a functioning wells file and a beadfind mask
// images are only known internally to this.
void GetFromImagesToWells (RawWells &rawWells, Mask *maskPtr,
                           CommandLineOpts &clo,
                           char *experimentName, string &analysisLocation,
                           int numFlows,
                           SeqListClass &my_keys,
                           TrackProgress &my_progress, Region &wholeChip,
                           int &well_rows, int &well_cols)
{

  /*
   *  Two ways to proceed.  If this is raw data analysis, continue from here.
   *  If this is a re-basecall of an existing wells file, jump to basecalling
   *  section.
   */
  if (!clo.mod_control.USE_RAWWELLS)
  {
    // do beadfind and signal processing
    RealImagesToWells (rawWells, maskPtr,clo,experimentName,analysisLocation,
                       numFlows,my_keys,my_progress,
                       wholeChip,well_rows,well_cols);

  } // end image processing
  else    // Processing --from-wells - no images are ever used
  {
    // Update progress bar file: well find is complete
    updateProgress (WELL_TO_IMAGE);

    // grab the previous beadfind
    LoadBeadMaskFromFile (clo.sys_context,maskPtr);
    SetSpatialContextAndMask(clo.loc_context, maskPtr, well_rows, well_cols);

    clo.WriteProcessParameters(); // because we won't have them, spill them all out

    // copy/link files from old directory for the report
    clo.sys_context.MakeSymbolicLinkToOldDirectory (experimentName);
    clo.sys_context.CopyFilesForReportGeneration (experimentName,my_keys);
    SetChipTypeFromWells (rawWells);
  }
}


