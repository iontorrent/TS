/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

#include "ProgramState.h"
#include "ChipIdDecoder.h"

ProgramState::ProgramState(std::string &path)
{ 
  if (path.empty())
    std::cerr << "[ProgramState] Error: No file name." << std::endl;
  else{
    outFile = path;
//    outFile += "analysisState.json";  
    std::cout << std::endl;    
  }
  // section names
  clo_name = "CommandLineOpts";
  mod_control_name = "ModuleControlOpts";
  key_context_name = "KeyContext";
  flow_context_name = "FlowContext";
  sys_context_name = "SystemContext"; 
  bkg_control_name = "BkgControlOpts"; 
  bfd_control_name = "BeadfindControlOpts"; 
  loc_context_name = "SpatialContext";
  img_control_name = "ImageControlOpts";
  seq_list_name = "SequenceList";
  img_spec_name = "ImageSpecClass";
  chip_id_name = "ChipID";
}

void ProgramState::WriteJson(const Json::Value & json, const std::string& filename_json)
{
  std::ofstream outJsonFile(filename_json.c_str(), std::ios::out);
  if (outJsonFile.good())
    outJsonFile << json.toStyledString();
  else
    std::cerr << "[ProgramState] Unable to write JSON file " << filename_json << std::endl;    
  outJsonFile.close();  
}

void ProgramState::LoadJson(Json::Value & json, const std::string& filename_json)
{
  std::ifstream inJsonFile(filename_json.c_str(), std::ios::in);
  if (inJsonFile.good())
    inJsonFile >> json;
  else
    std::cerr << "[ProgramState] Unable to read JSON file " << filename_json << std::endl;      
  inJsonFile.close();
}


// **** SAVE *****

void ProgramState::WriteState()
{
  std::cout << "[ProgramState] Saving program state to: " << outFile << std::endl;  
  WriteJson(state_json, outFile);
}

void ProgramState::Save(CommandLineOpts &clo, SeqListClass &seq, ImageSpecClass &my_image_spec)
{ 
  AddBkgControl(state_json[clo_name][bkg_control_name], clo.bkg_control);
  AddModControl(state_json[clo_name][mod_control_name], clo.mod_control); 
  AddSysContext(state_json[clo_name][sys_context_name], clo.sys_context); 
  AddKeyContext(state_json[clo_name][key_context_name], clo.key_context); 
  AddFlowContext(state_json[clo_name][flow_context_name], clo.flow_context);  
  AddBfdControl(state_json[clo_name][bfd_control_name], clo.bfd_control); 
  AddLocContext(state_json[clo_name][loc_context_name], clo.loc_context);
  AddImgControl(state_json[clo_name][img_control_name], clo.img_control);
  AddSeqList(state_json[seq_list_name], seq);
  AddImgSpec(state_json[img_spec_name], my_image_spec);
//  AddChipID(state_json[chip_id_name]);
}


void ProgramState::AddBkgControl(Json::Value &json,BkgModelControlOpts &bkg_control) 
{
  json["wellsCompression"] = bkg_control.signal_chunks.wellsCompression;
  json["flow_block_sequence"] = bkg_control.signal_chunks.flow_block_sequence.ToString();
  json["regional_smoothing_alpha"] = bkg_control.regional_smoothing.alpha;
  json["regional_smoothing_gamma"] = bkg_control.regional_smoothing.gamma;
}

void ProgramState::AddModControl(Json::Value &json, ModuleControlOpts &mod_control)
{
  json["passTau"] = mod_control.passTau;
}

void ProgramState::AddKeyContext(Json::Value &json, KeyContext &key_context)
{
  json["libKey"] = (key_context.libKey) ? key_context.libKey : "";
  json["tfKey"] = (key_context.tfKey) ? key_context.tfKey : "";
  json["maxNumKeyFlows"] = key_context.maxNumKeyFlows;
  json["minNumKeyFlows"] = key_context.minNumKeyFlows;
}

void ProgramState::AddFlowContext(Json::Value &json, FlowContext &flow_context)
{
  json["flowOrder"] = (flow_context.flowOrder) ? flow_context.flowOrder : "";
  json["flowOrderOverride"] = flow_context.flowOrderOverride;
//  json["flowOrderIndex"] = (flow_context.flowOrderIndex) ? flow_context.flowOrderIndex : 0; //**this will be an array?
  json["numFlowsPerCycle"] = flow_context.numFlowsPerCycle;
  json["flowLimitSet"] = flow_context.flowLimitSet;
  json["numTotalFlows"] = flow_context.numTotalFlows;  
  json["flow_range_set"] = flow_context.flow_range_set; 
  json["startingFlow"] = flow_context.startingFlow; 
  json["endingFlow"] = flow_context.endingFlow; 
}

void ProgramState::AddSysContext(Json::Value &json, SystemContext &sys_context)
{
  json["results_folder"] = (sys_context.results_folder) ? sys_context.results_folder : "";
  json["dat_source_directory"] = (sys_context.dat_source_directory) ? sys_context.dat_source_directory : "";
  json["wells_output_directory"] = (sys_context.wells_output_directory) ? sys_context.wells_output_directory : "";
  json["analysisLocation"] = sys_context.analysisLocation;
  json["wellsFileName"] = sys_context.wellsFileName;
  json["tmpWellsFile"] = sys_context.tmpWellsFile;
  json["runId"] = sys_context.runId;
  json["wellsFilePath"] = sys_context.wellsFilePath;
  json["wellStatFile"] = (sys_context.wellStatFile.length() > 0) ? sys_context.wellStatFile : "";
  json["NO_SUBDIR"] = sys_context.NO_SUBDIR;
  json["LOCAL_WELLS_FILE"] = sys_context.LOCAL_WELLS_FILE;
  json["wellsFormat"] = sys_context.wellsFormat;
  json["explog_path"] = (sys_context.explog_path.length() > 0) ? sys_context.explog_path : "";
}

void ProgramState::AddBfdControl(Json::Value &json, BeadfindControlOpts &bfd_control)
{
  json["bfMinLiveRatio"] = bfd_control.bfMinLiveRatio;
  json["bfMinLiveLibSnr"] = bfd_control.bfMinLiveLibSnr;
  json["bfMinLiveTfSnr"] = bfd_control.bfMinLiveTfSnr;
  json["bfTfFilterQuantile"] = bfd_control.bfTfFilterQuantile;
  json["bfLibFilterQuantile"] = bfd_control.bfLibFilterQuantile;
  json["skipBeadfindSdRecover"] = bfd_control.skipBeadfindSdRecover;
  json["beadfindThumbnail"] = bfd_control.beadfindThumbnail;
  json["filterNoisyCols"] = bfd_control.filterNoisyCols;
  json["beadfindSmoothTrace"] = bfd_control.beadfindSmoothTrace;
  json["beadMaskFile"] = (bfd_control.beadMaskFile) ? bfd_control.beadMaskFile : "";
  json["maskFileCategorized"] = bfd_control.maskFileCategorized;
  json["bfFileBase"] = bfd_control.bfFileBase;
  json["preRunbfFileBase"] = bfd_control.preRunbfFileBase;
  json["noduds"] = bfd_control.noduds;
  json["bfOutputDebug"] = bfd_control.bfOutputDebug;  
  json["beadfindType"] = bfd_control.beadfindType;
  json["bfType"] = bfd_control.bfType;
  json["bfDat"] = bfd_control.bfDat;
  json["bfBgDat"] = bfd_control.bfBgDat;
  json["SINGLEBF"] = bfd_control.SINGLEBF;
  json["BF_ADVANCED"] = bfd_control.BF_ADVANCED;
}

void ProgramState::AddLocContext(Json::Value &json, SpatialContext &loc_context)
{
  json["numRegions"] = loc_context.numRegions;
  json["cols"] = loc_context.cols;
  json["rows"] = loc_context.rows;
  json["regionXSize"] = loc_context.regionXSize;
  json["regionYSize"] = loc_context.regionYSize;
  json["regionsX"] = loc_context.regionsX;
  json["regionsY"] = loc_context.regionsY;
  json["numCropRegions"] = loc_context.numCropRegions;   
  for (int r = 0; r < loc_context.numCropRegions; r++) {      
    json["cropRegions"]["row"][r] = loc_context.cropRegions[r].row;
    json["cropRegions"]["col"][r] = loc_context.cropRegions[r].col;
    json["cropRegions"]["w"][r] = loc_context.cropRegions[r].w;
    json["cropRegions"]["h"][r] = loc_context.cropRegions[r].h;
    json["cropRegions"]["index"][r] = loc_context.cropRegions[r].index;
  }
  json["cropped_region_x_offset"] = loc_context.cropped_region_x_offset;
  json["cropped_region_y_offset"] = loc_context.cropped_region_y_offset;
  json["chip_offset_x"] = loc_context.chip_offset_x;
  json["chip_offset_y"] = loc_context.chip_offset_y;
  json["chip_len_x"] = loc_context.chip_len_x;
  json["chip_len_y"] = loc_context.chip_len_y;
  json["chipRegion"]["row"] = loc_context.chipRegion.row;
  json["chipRegion"]["col"] = loc_context.chipRegion.col;
  json["chipRegion"]["w"] = loc_context.chipRegion.w;
  json["chipRegion"]["h"] = loc_context.chipRegion.h;
  json["chipRegion"]["index"] = loc_context.chipRegion.index;  
  json["percentEmptiesToKeep"] = loc_context.percentEmptiesToKeep;
  json["exclusionMaskSet"] = loc_context.exclusionMaskSet;
}

void ProgramState::AddImgControl(Json::Value &json, ImageControlOpts &img_control)
{
  json["totalFrames"] = img_control.totalFrames;
  json["maxFrames"] = img_control.maxFrames;
  json["nn_subtract_empties"] = img_control.nn_subtract_empties;
  json["NNinnerx"] = img_control.NNinnerx;
  json["NNinnery"] = img_control.NNinnery;
  json["NNouterx"] = img_control.NNouterx;
  json["NNoutery"] = img_control.NNoutery;  
  json["hilowPixFilter"] = img_control.hilowPixFilter;
  json["ignoreChecksumErrors"] = img_control.ignoreChecksumErrors;
  json["flowTimeOffset"] = img_control.flowTimeOffset;  
  json["gain_correct_images"] = img_control.gain_correct_images;
  json["gain_debug_output"] = img_control.gain_debug_output;  
  json["outputPinnedWells"] = img_control.outputPinnedWells;
  json["tikSmoothingFile"] = img_control.tikSmoothingFile;
  json["tikSmoothingInternal"] = img_control.tikSmoothingInternal;
  json["total_timeout"] = img_control.total_timeout; 
  json["has_wash_flow"] = img_control.has_wash_flow; 
}

void ProgramState::AddSeqList(Json::Value &json, SeqListClass &seq)
{
  json["numSeqListItems"] = seq.numSeqListItems;
  for (int i = 0; i < seq.numSeqListItems; i++) { 
    json["type"][i] = seq.seqList[i].type;
    json["seq"][i] = seq.seqList[i].seq;
    json["numKeyFlows"][i] = seq.seqList[i].numKeyFlows;
    json["usableKeyFlows"][i] = seq.seqList[i].usableKeyFlows;
    for (int flow = 0; flow < seq.seqList[i].numKeyFlows; flow++){
      json["Ionogram"][i][flow] = seq.seqList[i].Ionogram[flow];
      json["zeromers"][i][flow] = seq.seqList[i].zeromers[flow];
      json["onemers"][i][flow] = seq.seqList[i].onemers[flow];
    } 
  }
}

void ProgramState::AddImgSpec(Json::Value &json, ImageSpecClass &my_image_spec)
{
  json["rows"] = my_image_spec.rows;
  json["cols"] = my_image_spec.cols;
  json["scale_of_chip"] = my_image_spec.scale_of_chip;
  json["uncompFrames"] = my_image_spec.uncompFrames;
//  for (int i = 0; i < img_control.maxFrames; i++) 
//    json["timestamps"][i] = my_image_spec.timestamps[i]; //save timestamps in hdf5
  json["vfr_enabled"] = my_image_spec.vfr_enabled;
}

void ProgramState::AddChipID(Json::Value &json)
{
//  json["chipType"] = ChipIdDecoder::GetChipType();
//  json["ChipIdEnum"] = ChipIdDecoder::GetGlobalChipId();
}

// **** LOAD *****

void ProgramState::LoadState(CommandLineOpts &clo, SeqListClass &seq, ImageSpecClass &my_image_spec)
{
  std::cout << "[ProgramState] Loading program state from: " << outFile << std::endl;  
  LoadJson(state_json, outFile);

  //SetSysContext(state_json[clo_name][sys_context_name], clo.sys_context);
  clo.SetSysContextLocations();
  //SetBkgControl(state_json[clo_name][bkg_control_name], clo.bkg_control);
  SetModControl(state_json[clo_name][mod_control_name], clo.mod_control);
  //SetKeyContext(state_json[clo_name][key_context_name], clo.key_context);
  // beadfind should not be setting this
  // SetFlowContext(state_json[clo_name][flow_context_name], clo.flow_context);
  clo.SetFlowContext(clo.sys_context.explog_path);
  SetLocContext(state_json[clo_name][loc_context_name], clo.loc_context);
  SetImgControl(state_json[clo_name][img_control_name], clo.img_control);
  SetSeqList(state_json[seq_list_name], seq);
  SetImgSpec(state_json[img_spec_name], my_image_spec);
  SetChipID(state_json[chip_id_name]);
}

void ProgramState::SetBkgControl(Json::Value &json, BkgModelControlOpts &bkg_control) 
{
  bkg_control.signal_chunks.wellsCompression = json["wellsCompression"].asUInt();
  bkg_control.signal_chunks.flow_block_sequence.Set( json["flow_block_sequence"].asCString() );
  bkg_control.regional_smoothing.alpha = json["regional_smoothing_alpha"].asFloat();
  bkg_control.regional_smoothing.gamma = json["regional_smoothing_gamma"].asFloat();
}

void ProgramState::SetModControl(Json::Value &json, ModuleControlOpts &mod_control)
{
  mod_control.BEADFIND_ONLY = 0;
  mod_control.passTau = json["passTau"].asBool();
  mod_control.reusePriorBeadfind = true; 
}

void ProgramState::SetFlowContext(Json::Value &json, FlowContext &flow_context)
{
  flow_context.flowOrder = strdup(json["flowOrder"].asString().c_str());  
  flow_context.flowOrderOverride = json["flowOrderOverride"].asBool();  
//  flow_context.flowOrderIndex =  //**this will be an array?
  flow_context.numFlowsPerCycle = json["numFlowsPerCycle"].asUInt();
//don't want to overwrite cmd-line options for the flows processed
  flow_context.flowLimitSet = (flow_context.flowLimitSet) ? flow_context.flowLimitSet : json["flowLimitSet"].asUInt();
  flow_context.numTotalFlows = (flow_context.flowLimitSet) ? flow_context.flowLimitSet : json["numTotalFlows"].asUInt(); 
//  flow_context.flow_range_set = json["flow_range_set"].asBool(); 
  flow_context.startingFlow = (flow_context.flow_range_set) ? flow_context.startingFlow : json["startingFlow"].asUInt(); 
  flow_context.endingFlow = (flow_context.flow_range_set) ? flow_context.endingFlow : json["endingFlow"].asUInt();  
}

void ProgramState::SetKeyContext(Json::Value &json, KeyContext &key_context)
{
  key_context.libKey = strdup(json["libKey"].asString().c_str());
  key_context.tfKey = strdup(json["tfKey"].asString().c_str());
  key_context.maxNumKeyFlows = json["maxNumKeyFlows"].asInt();
  key_context.minNumKeyFlows = json["minNumKeyFlows"].asInt();
}

void ProgramState::SetSysContext(Json::Value &json, SystemContext &sys_context)
{  
  // allow us to change directories and control wells files
  // sys_context.results_folder = strdup(json["results_folder"].asString().c_str());
  // sys_context.dat_source_directory = strdup(json["dat_source_directory"].asString().c_str());
  // sys_context.wells_output_directory = strdup(json["wells_output_directory"].asString().c_str());
  // sys_context.analysisLocation = json["analysisLocation"].asString();
  strcpy(sys_context.wellsFileName, json["wellsFileName"].asString().c_str());
  strcpy(sys_context.tmpWellsFile, json["tmpWellsFile"].asString().c_str());
  strcpy(sys_context.runId, json["runId"].asString().c_str());
  strcpy(sys_context.wellsFilePath, json["wellsFilePath"].asString().c_str());    
  sys_context.wellStatFile = strdup(json["wellStatFile"].asString().c_str());  
  // sys_context.NO_SUBDIR = json["NO_SUBDIR"].asInt();
  sys_context.LOCAL_WELLS_FILE = json["LOCAL_WELLS_FILE"].asInt();
  sys_context.wellsFormat = json["wellsFormat"].asString();
  // sys_context.explog_path = strdup(json["explog_path"].asString().c_str());
  sys_context.FindExpLogPath();
}

void ProgramState::SetLocContext(Json::Value &json, SpatialContext &loc_context)
{
  loc_context.numRegions = json["numRegions"].asInt();
  loc_context.cols = json["cols"].asInt();
  loc_context.rows = json["rows"].asInt();
  if ( loc_context.IsSetRegionXYSize() ){
    int regionXSize = json["regionXSize"].asInt();
    int regionYSize = json["regionYSize"].asInt();
    if ( (loc_context.regionXSize != regionXSize) ||
	 (loc_context.regionYSize != regionYSize))
      fprintf(stdout, "Warning: option --region-size=%dx%d ignored, using %dx%d\n", loc_context.regionXSize, loc_context.regionYSize,regionXSize, regionYSize);    
  }
  loc_context.regionXSize = json["regionXSize"].asInt();
  loc_context.regionYSize = json["regionYSize"].asInt();
  loc_context.regionsX = json["regionsX"].asInt();
  loc_context.regionsY = json["regionsY"].asInt();
  loc_context.numCropRegions = json["numCropRegions"].asInt(); 
      
  loc_context.cropRegions = (Region *) realloc (loc_context.cropRegions, sizeof(Region) * loc_context.numCropRegions);    
  for (int r = 0; r < loc_context.numCropRegions; r++) {          
    loc_context.cropRegions[r].row = json["cropRegions"]["row"][r].asInt();
    loc_context.cropRegions[r].col = json["cropRegions"]["col"][r].asInt();
    loc_context.cropRegions[r].w = json["cropRegions"]["w"][r].asInt();
    loc_context.cropRegions[r].h = json["cropRegions"]["h"][r].asInt();
    loc_context.cropRegions[r].index = json["cropRegions"]["index"][r].asInt();
    loc_context.isCropped = true;
  }
  
  loc_context.cropped_region_x_offset = json["cropped_region_x_offset"].asInt();
  loc_context.cropped_region_y_offset = json["cropped_region_y_offset"].asInt();
  loc_context.chip_offset_x = json["chip_offset_x"].asInt();
  loc_context.chip_offset_y = json["chip_offset_y"].asInt();
  loc_context.chip_len_x = json["chip_len_x"].asInt();
  loc_context.chip_len_y = json["chip_len_y"].asInt();
  loc_context.chipRegion.row = json["chipRegion"]["row"].asInt();
  loc_context.chipRegion.col = json["chipRegion"]["col"].asInt();
  loc_context.chipRegion.w = json["chipRegion"]["w"].asInt();
  loc_context.chipRegion.h = json["chipRegion"]["h"].asInt();
  loc_context.chipRegion.index = json["chipRegion"]["index"].asInt();
  loc_context.percentEmptiesToKeep = json["percentEmptiesToKeep"].asDouble();
  loc_context.exclusionMaskSet = json["exclusionMaskSet"].asBool();
  
  // Also set ImageCropping static variables 
  ImageCropping::SetCroppedSubRegion(loc_context.chipRegion);
  ImageCropping::SetCroppedRegionOrigin(loc_context.cropped_region_x_offset, loc_context.cropped_region_y_offset);
  
}

void ProgramState::SetImgControl(Json::Value &json, ImageControlOpts &img_control)
{
  img_control.totalFrames = json["totalFrames"].asInt();
  img_control.maxFrames = json["maxFrames"].asInt();
  img_control.nn_subtract_empties = json["nn_subtract_empties"].asInt();
  img_control.NNinnerx = json["NNinnerx"].asInt();
  img_control.NNinnery = json["NNinnery"].asInt();
  img_control.NNouterx = json["NNouterx"].asInt();
  img_control.NNoutery = json["NNoutery"].asInt();
  img_control.hilowPixFilter = json["hilowPixFilter"].asInt();
  img_control.ignoreChecksumErrors = json["ignoreChecksumErrors"].asInt();
  img_control.flowTimeOffset = json["flowTimeOffset"].asInt();
  img_control.gain_correct_images = json["gain_correct_images"].asBool();
  img_control.gain_debug_output = json["gain_debug_output"].asBool();
  img_control.outputPinnedWells = json["outputPinnedWells"].asInt();
  strcpy(img_control.tikSmoothingFile, json["tikSmoothingFile"].asString().c_str());
  strcpy(img_control.tikSmoothingInternal, json["tikSmoothingInternal"].asString().c_str());
  img_control.total_timeout = json["total_timeout"].asInt(); 
  img_control.has_wash_flow = json["has_wash_flow"].asInt();
}  

void ProgramState::SetSeqList(Json::Value &json, SeqListClass &seq)
{
  seq.numSeqListItems = json["numSeqListItems"].asInt();
  seq.seqList = new SequenceItem[seq.numSeqListItems];
  for (int i = 0; i < seq.numSeqListItems; i++) {       
    seq.seqList[i].type = (json["type"][i] == MaskLib ? MaskLib : MaskTF);       
    seq.seqList[i].seq = strdup(json["seq"][i].asString().c_str());
    seq.seqList[i].numKeyFlows = json["numKeyFlows"][i].asInt();
    seq.seqList[i].usableKeyFlows = json["usableKeyFlows"][i].asInt();
    for (int flow = 0; flow < seq.seqList[i].numKeyFlows; flow++){
      seq.seqList[i].Ionogram[flow] = json["Ionogram"][i][flow].asInt();
      seq.seqList[i].zeromers[flow] = json["zeromers"][i][flow].asInt();
      seq.seqList[i].onemers[flow] = json["onemers"][i][flow].asInt();
    }
  }
}

void ProgramState::SetImgSpec(Json::Value &json, ImageSpecClass &my_image_spec)
{
  my_image_spec.rows = json["rows"].asInt();
  my_image_spec.cols = json["cols"].asInt();
  my_image_spec.scale_of_chip = json["scale_of_chip"].asInt();
  my_image_spec.uncompFrames = json["uncompFrames"].asUInt();
  my_image_spec.vfr_enabled = json["vfr_enabled"].asBool();
}

void ProgramState::SetChipID(Json::Value &json)
{
  ChipIdDecoder::SetGlobalChipId(json["chipType"].asString().c_str());
}
