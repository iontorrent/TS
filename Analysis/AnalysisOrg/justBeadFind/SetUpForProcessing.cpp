/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SetUpForProcessing.h"
#include <fstream>

void SetUpKeys(SeqListClass &my_keys, KeyContext &key_context, FlowContext &flow_context)
{
  my_keys.StdInitialize (flow_context.flowOrder,key_context.libKey, key_context.tfKey); // 8th duplicated flow processing code
  //@TODO: these parameters are just for reporting purposes???
  // they appear to be ignored everywhere
  my_keys.UpdateMinFlows (key_context.minNumKeyFlows);
  my_keys.UpdateMaxFlows (key_context.maxNumKeyFlows);
}

//@TODO: bad coding practice to have side effects like this does in the static parameters of a model
// this is just a way of disguising global variables
void SetUpToProcessImages ( ImageSpecClass &my_image_spec, CommandLineOpts &inception_state )
{
  ImageCropping::SetCroppedSubRegion ( inception_state.loc_context.chipRegion );

  // make sure we're using XTCorrection at the right offset for cropped regions
  ImageCropping::SetCroppedRegionOrigin ( inception_state.loc_context.cropped_region_x_offset,inception_state.loc_context.cropped_region_y_offset );

  ImageTransformer::CalibrateChannelXTCorrection ( inception_state.sys_context.dat_source_directory,"lsrowimage.dat" );

  //Again smuggling "global" variables using static variables for side effects
  if ( inception_state.img_control.gain_correct_images )
    ImageTransformer::CalculateGainCorrectionFromBeadfindFlow ( inception_state.sys_context.dat_source_directory,inception_state.img_control.gain_debug_output );

  //@TODO: this mess has nasty side effects on the arguments.
  my_image_spec.DeriveSpecsFromDat ( inception_state.sys_context, inception_state.img_control, inception_state.loc_context ); // dummy - only reads 1 dat file

}

void SetUpOrLoadInitialState(CommandLineOpts &inception_state, SeqListClass &my_keys, TrackProgress &my_progress, ImageSpecClass &my_image_spec, SlicedPrequel& my_prequel_setup)
{
  if ( !inception_state.bkg_control.restart_from.empty() )
  {
    // restarting from saved computational state
    // beadfind, bfmask.bin and beadfind.h5 will be ignored

    // note that if we are here we will never load the separator data
    inception_state.sys_context.GenerateContext (); // find our directories
    inception_state.sys_context.SetUpAnalysisLocation();

    LoadBeadFindState(inception_state, my_keys, my_image_spec);

    inception_state.SetProtonDefault();
  }
  else if (inception_state.mod_control.reusePriorBeadfind && inception_state.bkg_control.restart_from.empty())
  {
    // starting execution fresh, justBeadFind already run
    
    // get any state from beadFind
    LoadBeadFindState(inception_state, my_keys, my_image_spec);

    inception_state.SetProtonDefault();

    // region layout saved in inception_state.loc_context
    // region definitions in background model via my_prequel_setup
    my_prequel_setup.SetRegions ( inception_state.loc_context.numRegions,
				  my_image_spec.rows,my_image_spec.cols,
				  inception_state.loc_context.regionXSize,
				  inception_state.loc_context.regionYSize );
    my_prequel_setup.FileLocations ( inception_state.sys_context.analysisLocation );

  }
 else
 {  
   // starting execution fresh, justBeadFind not run
   inception_state.SetUpProcessing();

   CreateResultsFolder (inception_state.sys_context.GetResultsFolder());
   inception_state.sys_context.SetUpAnalysisLocation();
    
   // convert from old key representatino to more useful modern style  
   SetUpKeys(my_keys, inception_state.key_context, inception_state.flow_context);
  
   //@TODO: side effects here on the entire image class
   // after this point, Images will behave differently when read in
   SetUpToProcessImages ( my_image_spec, inception_state );

   // region layout saved into inception_state.loc_context
   SetUpRegionsForAnalysis ( my_image_spec.rows, my_image_spec.cols, inception_state.loc_context );

   // region layout shared in background model and beadfind via my_prequel_setup
   my_prequel_setup.SetRegions ( inception_state.loc_context.numRegions,
				 my_image_spec.rows,my_image_spec.cols,
				 inception_state.loc_context.regionXSize,
				 inception_state.loc_context.regionYSize );
   my_prequel_setup.FileLocations ( inception_state.sys_context.analysisLocation );
 }
  fprintf(stdout, "Analysis region size is width %d, height %d\n", inception_state.loc_context.regionXSize, inception_state.loc_context.regionYSize);
}

void LoadBeadFindState(CommandLineOpts &inception_state, SeqListClass &my_keys, ImageSpecClass &my_image_spec)
{ 
  inception_state.sys_context.GenerateContext (); // find our directories
  inception_state.sys_context.SetUpAnalysisLocation();
  std::string analysisLocation = inception_state.sys_context.analysisLocation;
  std::string stateFile =  analysisLocation + "/analysisState.json";
  std::string imageSpecFile =  analysisLocation + "/imageState.h5";
      
  if (isFile (stateFile.c_str())){    
    // load state parameters    
    ProgramState state(stateFile);  
    state.LoadState(inception_state,my_keys,my_image_spec);  
    // load image state
    CaptureImageState imgState( imageSpecFile );
    imgState.LoadImageSpec(my_image_spec);
    imgState.LoadXTCorrection();
    if (inception_state.img_control.gain_correct_images)
      imgState.LoadImageGainCorrection(my_image_spec.rows, my_image_spec.cols);
  }
  else
  {   
    fprintf (stderr, "Unable to find state parameter file %s\n", stateFile.c_str());
    exit (EXIT_FAILURE);
  }  
}


