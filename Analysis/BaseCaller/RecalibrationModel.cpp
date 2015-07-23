/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     RecalibrationModel.cpp
//! @ingroup  BaseCaller
//! @brief    RecalibrationModel. Model estimation between simulated predictions and observed measurements

#include "RecalibrationModel.h"


int NuctoInt(char nuc) {
    switch (nuc) {
    case 'A':
        return 0;
    case 'C':
        return 1;
    case 'G':
        return 2;
    default:
        return 3;
    }
}

RecalibrationModel::RecalibrationModel()
{
    is_enabled_ = false;
    max_hp_calibrated_ = 0;
    recalModelHPThres = 4;
    verbose_= true;
}


RecalibrationModel::~RecalibrationModel()
{
}

int rGetParamsInt(Json::Value& json, const string& key, int default_value) {
    if (not json.isMember(key))
        return default_value;
    if (json[key].isString())
        return atoi(json[key].asCString());
    return json[key].asInt();
}

double rGetParamsDbl(Json::Value& json, const string& key, double default_value) {
    if (not json.isMember(key))
        return default_value;
    if (json[key].isString())
        return atof(json[key].asCString());
    return json[key].asDouble();
}


void RecalibrationModel::InitializeFromJSON(Json::Value &recal_params, string &my_block_key, bool spam_enabled, int over_flow_protect) {
  // this needs to signal when it fails in some way
	verbose_ = spam_enabled;
  
    int flowStart, flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan, max_hp_calibrated;
    flowStart = rGetParamsInt(recal_params[my_block_key],"flowStart",0);
    //cout << flowStart << endl;
    flowEnd = rGetParamsInt(recal_params[my_block_key],"flowEnd",0);
    if (over_flow_protect>flowEnd)
      flowEnd = over_flow_protect;  // allocate for combined bam files that need to access flows "past the end" without crashing
    //cout << flowEnd << endl;
    flowSpan = rGetParamsInt(recal_params[my_block_key],"flowSpan",0);
    //cout << flowSpan << endl;
    xMin = rGetParamsInt(recal_params[my_block_key],"xMin",0);
    //cout << xMin << endl;
    xMax =rGetParamsInt(recal_params[my_block_key],"xMax",0);
    //cout << xMax << endl;
    xSpan =rGetParamsInt(recal_params[my_block_key],"xSpan",0);
    //cout << xSpan << endl;
    yMin = rGetParamsInt(recal_params[my_block_key],"yMin",0);
    //cout << yMin << endl;
    yMax =rGetParamsInt(recal_params[my_block_key],"yMax",0);
    //cout << yMax << endl;
    ySpan = rGetParamsInt(recal_params[my_block_key],"ySpan",0);
    //cout << ySpan << endl;
    max_hp_calibrated = rGetParamsInt(recal_params[my_block_key],"max_hp_calibrated",0);
    stratification.SetupChipRegions(xMin, xMax, xSpan, yMin, yMax, ySpan);
    stratification.SetupFlowRegions(flowStart,flowEnd, flowSpan);
    //calculate number of partitions and initialize the stratifiedAs and stratifiedBs
    SetupStratification(flowStart,flowEnd, flowSpan,xMin,xMax,xSpan,yMin,yMax,ySpan,max_hp_calibrated);

    // stratification setup done
    // now iterate and obtain each line from the JSON
    int iter_size = recal_params[my_block_key]["modelParameters"].size();
    for (int i_item=0; i_item<iter_size; i_item++) {
        // extract my single line
        //model_file >> flowBase >> flowStart >> flowEnd >> xMin >> xMax >> yMin >> yMax >> refHP >> paramA >> paramB;
        //flowBase is a special extraction
        flowStart = rGetParamsInt(recal_params[my_block_key]["modelParameters"][i_item],"flowStart",0);
        flowEnd = rGetParamsInt(recal_params[my_block_key]["modelParameters"][i_item],"flowEnd",0);
        xMin = rGetParamsInt(recal_params[my_block_key]["modelParameters"][i_item],"xMin",0);
        xMax =rGetParamsInt(recal_params[my_block_key]["modelParameters"][i_item],"xMax",0);
        yMin = rGetParamsInt(recal_params[my_block_key]["modelParameters"][i_item],"yMin",0);
        yMax =rGetParamsInt(recal_params[my_block_key]["modelParameters"][i_item],"yMax",0);

        int refHP;
        refHP = rGetParamsInt(recal_params[my_block_key]["modelParameters"][i_item],"refHP",0);
        float paramA, paramB;
        paramA = rGetParamsDbl(recal_params[my_block_key]["modelParameters"][i_item],"paramA",1.0);
        paramB = rGetParamsDbl(recal_params[my_block_key]["modelParameters"][i_item],"paramB",0.0);

        //string flowBase = recal_params[my_block_key]["modelParameters"][i_item]["flowBase"].asCString();
        char flowBase = (char) rGetParamsInt(recal_params[my_block_key]["modelParameters"][i_item],"flowBase",0);
        int nucInd = NuctoInt(flowBase);
        
        // all set with the values
        int offsetRegion = stratification.OffsetRegion(xMin,yMin);
        // note we only fill >in< flows fit by the recalibration model
        FillIndexes(offsetRegion,nucInd, refHP, flowStart, flowEnd, paramA, paramB);        
    }
    // if something bad happened above, how do we find out? - We do some basic sanity checks
    CheckArraySize();
    // now we're done!
    if (verbose_)
      cout << "Recalibration: enabled (using recalibration comment " << my_block_key << ") in a "
           << stratification.xCuts << 'x' << stratification.yCuts << 'x' << stratification.flowCuts << " grid." <<endl << endl;
    is_enabled_ = true;
}

void RecalibrationModel::SetupStratification(int flowStart, int flowEnd, int flowSpan,
        int xMin, int xMax, int xSpan,
        int yMin, int yMax, int ySpan, int max_hp_calibrated) {
    const int numRegionStratifications = stratification.xCuts * stratification.yCuts;
    const int numFlows = flowEnd - flowStart + 1;
    const int numHPs = MAX_HPXLEN + 1; //max_hp_calibrated + 1;
    const int numNucs = 4;
    stratifiedAs.resize(numRegionStratifications);
    stratifiedBs.resize(numRegionStratifications);
    for (int ind = 0; ind < numRegionStratifications; ++ind) {
        stratifiedAs.at(ind).resize(numFlows);
        stratifiedBs.at(ind).resize(numFlows);
        for (int flowInd = 0; flowInd < numFlows; flowInd++) {
            stratifiedAs.at(ind).at(flowInd).resize(numNucs);
            stratifiedBs.at(ind).at(flowInd).resize(numNucs);
            for (int nucInd = 0; nucInd < numNucs; ++nucInd) {
                stratifiedAs.at(ind).at(flowInd).at(nucInd).assign(numHPs, 1.0);
                stratifiedBs.at(ind).at(flowInd).at(nucInd).assign(numHPs, 0.0);
            }
        }
    }
}

// --------------------------------------------------------------------

void RecalibrationModel::Initialize(OptArgs& opts, vector<string> &bam_comments, const string & run_id, const ion::ChipSubset & chip_subset)
{
  string model_file_name    = opts.GetFirstString ('-', "model-file", "");
  int model_threshold       = opts.GetFirstInt('-', "recal-model-hp-thres", 4);
  bool save_hpmodel         = opts.GetFirstBoolean('-', "save-hpmodel", true);
  bool diagonal_state_prog  = opts.GetFirstBoolean('-', "diagonal-state-prog", false);

  if (diagonal_state_prog)
    model_file_name.clear();

  if (InitializeModel(model_file_name, model_threshold) and save_hpmodel)
    SaveModelFileToBamComments(model_file_name, bam_comments, run_id, chip_subset.GetColOffset(), chip_subset.GetRowOffset());
}

// --------------------------------------------------------------------

bool RecalibrationModel::InitializeModel(string model_file_name, int model_threshold)
{
    is_enabled_ = false;

    if (model_file_name.empty() or model_file_name == "off") {
        printf("RecalibrationModel: disabled\n\n");
        return false;
    }

    ifstream model_file;
    model_file.open(model_file_name.c_str());
    if (model_file.fail()) {
        printf("RecalibrationModel: disabled (cannot open %s)\n\n", model_file_name.c_str());
        model_file.close();
        return false;
    }

    if (model_threshold < 0 or model_threshold > MAX_HPXLEN) {
      cout << "RecalibrationModel: disabled (invalid model threshold of "<< model_threshold <<")" << endl;
      return false;
    } else
      recalModelHPThres = model_threshold;

    string comment_line;
    getline(model_file, comment_line); //skip the comment time

    int flowStart, flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan, max_hp_calibrated;
    model_file >> flowStart >> flowEnd >> flowSpan >> xMin >> xMax >> xSpan >> yMin >> yMax >> ySpan >>  max_hp_calibrated;
    stratification.SetupChipRegions(xMin, xMax, xSpan, yMin, yMax, ySpan);
    stratification.SetupFlowRegions(flowStart,flowEnd, flowSpan);
    //calculate number of partitions and initialize the stratifiedAs and stratifiedBs
    SetupStratification(flowStart,flowEnd, flowSpan,xMin,xMax,xSpan,yMin,yMax,ySpan,max_hp_calibrated);

    //TODO: parse model_file into stratifiedAs and stratifiedBs
    while (model_file.good()) {
        float paramA, paramB;
        int refHP;
        char flowBase;
        model_file >> flowBase >> flowStart >> flowEnd >> xMin >> xMax >> yMin >> yMax >> refHP >> paramA >> paramB;
        //populate it to stratifiedAs and startifiedBs
        int offsetRegion = stratification.OffsetRegion(xMin,yMin);
        FillIndexes(offsetRegion, NuctoInt(flowBase), refHP, flowStart, flowEnd, paramA, paramB);
    }

    model_file.close();

    cout << "Recalibration Model: enabled (using calibration file " << model_file_name << ")" << endl;
    cout << " - using calibration model for HPs " << recalModelHPThres << " and up in a "
           << stratification.xCuts << 'x' << stratification.yCuts << 'x' << stratification.flowCuts << " grid." << endl << endl;
    is_enabled_ = true;
    return is_enabled_;
}

void RecalibrationModel::FillIndexes(int offsetRegion, int nucInd, int refHP, int flowStart, int flowEnd, float paramA, float paramB) {
    for (int flowInd = flowStart; flowInd < flowEnd; ++flowInd) {
        if (refHP < recalModelHPThres) continue;
        stratifiedAs.at(offsetRegion).at(flowInd).at(nucInd).at(refHP) = paramA;
        stratifiedBs.at(offsetRegion).at(flowInd).at(nucInd).at(refHP) = paramB;
    }
}

void RecalibrationModel::getAB(MultiAB &multi_ab, int x, int y) const
{
     if (!is_enabled_) {
       multi_ab.Null();
    }
    int offsetRegion = stratification.OffsetRegion(x,y);
    if (offsetRegion < 0 || offsetRegion >= (int)stratifiedAs.size() || offsetRegion>=(int)stratifiedBs.size())
    {
        if (verbose_)
    	    cerr << "RecalibrationModel::getAB ERROR: offsetRegion "
                 << offsetRegion << " out of bounds " << stratifiedAs.size() << ',' << stratifiedBs.size()
                 << " for well x=" << x << " y=" << y << endl;
    	multi_ab.Null();
    }
    else{
      multi_ab.aPtr = &(stratifiedAs[offsetRegion]);
      multi_ab.bPtr = &(stratifiedBs[offsetRegion]);
    }
}

const vector<vector<vector<float> > > * RecalibrationModel::getAs(int x, int y) const
{
    if (!is_enabled_) {
        return 0;
    }
    int offsetRegion = stratification.OffsetRegion(x,y);
    if (offsetRegion < 0 || offsetRegion >= (int)stratifiedAs.size()) {
    	if (verbose_)
            cerr << "RecalibrationModel::getAs ERROR: offsetRegion "
    	         << offsetRegion << " out of bounds " << stratifiedAs.size()
    	         << " for well x=" << x << " y=" << y << endl;
        return 0;
    }
    else
        return &(stratifiedAs[offsetRegion]);
}

const vector<vector<vector<float> > > * RecalibrationModel::getBs(int x, int y) const
{
    if (!is_enabled_) {
        return 0;
    }
    int offsetRegion = stratification.OffsetRegion(x,y);
    if (offsetRegion < 0 || offsetRegion >= (int)stratifiedBs.size()) {
    	if (verbose_)
    	     cerr << "RecalibrationModel::getBs ERROR: offsetRegion "
                  << offsetRegion << " out of bounds " << stratifiedBs.size()
                  << " for well x=" << x << " y=" << y << endl;
        return 0;
    }
    else
        return &(stratifiedBs[offsetRegion]);
}

// ----------------------------------------------------------------
void RecalibrationModel::SaveModelFileToBamComments(string model_file_name, vector<string> &comments, const string &run_id, int block_col_offset, int block_row_offset)
{

    if (!model_file_name.empty())
    {
        ifstream model_file;
        model_file.open(model_file_name.c_str());
        if (!model_file.fail())
        {
            Json::Value hpJson(Json::objectValue);

            char buf[1000];
            string id = run_id;

            sprintf(buf, ".block_X%d_Y%d", block_col_offset, block_row_offset);
            id += buf;
            hpJson["MagicCode"] = "6d5b9d29ede5f176a4711d415d769108"; // md5hash "This uniquely identifies json comments for recalibration."
            hpJson["MasterKey"] = id;
            hpJson["MasterCol"] = block_col_offset;
            hpJson["MasterRow"] = block_row_offset;

            string comment_line;
            getline(model_file, comment_line); //skip the comment time

            int flowStart, flowEnd, flowSpan, xMin, xMax, xSpan, yMin, yMax, ySpan, max_hp_calibrated;
            model_file >> flowStart >> flowEnd >> flowSpan >> xMin >> xMax >> xSpan >> yMin >> yMax >> ySpan >> max_hp_calibrated;
            hpJson[id]["flowStart"] = flowStart;
            hpJson[id]["flowEnd"] = flowEnd;
            hpJson[id]["flowSpan"] = flowSpan;
            hpJson[id]["xMin"] = xMin;
            hpJson[id]["xMax"] = xMax;
            hpJson[id]["xSpan"] = xSpan;
            hpJson[id]["yMin"] = yMin;
            hpJson[id]["yMax"] = yMax;
            hpJson[id]["ySpan"] = ySpan;
            hpJson[id]["max_hp_calibrated"] = max_hp_calibrated;

            char flowBase;
            int refHP;
            float paramA, paramB;
            int item = 0;
            while (model_file.good())
            {
                model_file >> flowBase >> flowStart >> flowEnd >> xMin >> xMax >> yMin >> yMax >> refHP >> paramA >> paramB;
                hpJson[id]["modelParameters"][item]["flowBase"] = flowBase;
                hpJson[id]["modelParameters"][item]["flowStart"] = flowStart;
                hpJson[id]["modelParameters"][item]["flowEnd"] = flowEnd;
                hpJson[id]["modelParameters"][item]["xMin"] = xMin;
                hpJson[id]["modelParameters"][item]["xMax"] = xMax;
                hpJson[id]["modelParameters"][item]["yMin"] = yMin;
                hpJson[id]["modelParameters"][item]["yMax"] = yMax;
                hpJson[id]["modelParameters"][item]["refHP"] = refHP;
                hpJson[id]["modelParameters"][item]["paramA"] = paramA;
                hpJson[id]["modelParameters"][item]["paramB"] = paramB;
                ++item;
            }

            model_file.close();
            Json::FastWriter writer;
            string str = writer.write(hpJson);
            // trim unwanted newline added by writer
            int last_char = str.size()-1;
            if (last_char>=0) {
                if (str[last_char]=='\n'){
                    str.erase(last_char,1);
                }
            }
            comments.push_back(str);
        } else {
          cerr << "RecalModel: Failed to save hp model file " << model_file_name  << " to BAM comments."<< endl;
        }
    }
}

// ----------------------------------------------------------------

// Returning array sizes and doing sanity checks on constructed arrays!
const vector<unsigned int> RecalibrationModel::CheckArraySize() const {

  vector<unsigned int> my_size(4,0);
  my_size[0] = stratifiedAs.size();
  if (stratifiedBs.size() != my_size[0]) {
    cerr << "ERROR in RecalibrationModel: stratifiedAs and stratifiedBs are different in size!"
         << stratifiedAs.size() << " <-> " << stratifiedBs.size() <<endl;
    exit(EXIT_FAILURE);
  }

  // 1) Loop over 'offset_regions' to check sizes
  for (unsigned int or_idx=0; or_idx < my_size[0]; ++or_idx){
    if (or_idx == 0)
      my_size[1] = stratifiedAs.at(0).size();

    if (stratifiedAs.at(or_idx).size() != my_size[1]) {
      cerr << "ERROR in RecalibrationModel: stratifiedAs[0].size() and stratifiedAs[" << or_idx << "].size() are different!"
           << stratifiedAs.at(0).size() << " <-> " << stratifiedAs.at(or_idx).size() <<endl;
      exit(EXIT_FAILURE);
    }
    if (stratifiedBs.at(or_idx).size() != my_size[1]) {
      cerr << "ERROR in RecalibrationModel: stratifiedAs[0].size() and stratifiedBs[" << or_idx << "].size() are different!"
           << stratifiedAs.at(0).size() << " <-> " << stratifiedBs.at(or_idx).size() <<endl;
      exit(EXIT_FAILURE);
       }
    // 2) Loop over 'flows' to check sizes
    for (unsigned int f_idx=0; f_idx < my_size[1]; ++f_idx){
      if (f_idx == 0)
        my_size[2] = stratifiedAs.at(0).at(0).size();
      if (stratifiedAs.at(or_idx).at(f_idx).size() != my_size[2]) {
        cerr << "ERROR in RecalibrationModel: stratifiedAs[0][0].size() and stratifiedAs[" << or_idx << "]["<< f_idx << "].size() are different! "
             << stratifiedAs.at(0).at(0).size() << " <-> " << stratifiedAs.at(or_idx).at(f_idx).size() <<endl;
        exit(EXIT_FAILURE);
      }
      if (stratifiedBs.at(or_idx).at(f_idx).size() != my_size[2]) {
        cerr << "ERROR in RecalibrationModel: stratifiedAs[0][0].size() and stratifiedBs[" << or_idx << "]["<< f_idx << "].size() are different! "
             << stratifiedAs.at(0).at(0).size() << " <-> " << stratifiedBs.at(or_idx).at(f_idx).size() <<endl;
        exit(EXIT_FAILURE);
      }

      // 3) Loop over 'nucs' to check sizes
      for (unsigned int n_idx=0; n_idx < my_size[2]; ++n_idx){
        if (n_idx == 0)
          my_size[3] = stratifiedAs.at(0).at(0).at(0).size();
        if (stratifiedAs.at(or_idx).at(f_idx).at(n_idx).size() != my_size[3]) {
          cerr << "ERROR in RecalibrationModel: stratifiedAs[0][0][0].size() and stratifiedAs[" << or_idx << "]["<< f_idx << "][" << n_idx << "].size() are different! "
               << stratifiedAs.at(0).at(0).at(0).size() << " <-> " << stratifiedAs.at(or_idx).at(f_idx).at(n_idx).size() <<endl;
          exit(EXIT_FAILURE);
        }
        if (stratifiedAs.at(or_idx).at(f_idx).at(n_idx).size() != my_size[3]) {
          cerr << "ERROR in RecalibrationModel: stratifiedAs[0][0][0].size() and stratifiedAs[" << or_idx << "]["<< f_idx << "][" << n_idx << "].size() are different! "
               << stratifiedAs.at(0).at(0).at(0).size() << " <-> " << stratifiedAs.at(or_idx).at(f_idx).at(n_idx).size() <<endl;
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  return my_size;
}



