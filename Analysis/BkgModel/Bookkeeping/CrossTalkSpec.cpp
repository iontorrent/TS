/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include "CrossTalkSpec.h"
#include "Utils.h"

using namespace std;

// set up cross-talk information in a clean way

void TraceCrossTalkSpecification::Allocate(int size)
{
    nei_affected = size;
    cx.resize(size);
    cy.resize(size);
    tau_top.resize(size);
    tau_fluid.resize(size);
    multiplier.resize(size);
}

TraceCrossTalkSpecification::TraceCrossTalkSpecification()
{
    nei_affected = 0;

    hex_packed = false;
    three_series = true;
    initial_phase = 0;
    do_xtalk_correction = true;
    simple_model = false;
    rescale_flag = false;
    if_block_analysis = false;
    full_chip_x = 0;
    full_chip_y = 0;
    chipType = "";
    chipType_loaded = "";
}

// modified json format to support per-block parameters
/*
void TraceCrossTalkSpecification::PackTraceXtalkInfo(Json::Value &json)
{
  json["NeiAffected"] = nei_affected;
  for (int mi=0; mi<nei_affected; mi++){
    // for each neighbor
    json["CX"][mi] = cx[mi];
    json["CY"][mi] = cy[mi];
    json["TauTop"][mi] = tau_top[mi];
    json["TauFluid"][mi] = tau_fluid[mi];
    json["Multiplier"][mi] = multiplier[mi];
  }
  // grid specification is "complicated"
  json["OldGrid"]["hexpacked"] = hex_packed ? 1:0;
  json["OldGrid"]["three_series"] = three_series? 1:0;
  json["OldGrid"]["initial_phase"] = initial_phase;

  json["Xtype"] = simple_model ? "simple": "complex";
  json["Rescale"] = rescale_flag ? "rescale": "noscale";

}

void TraceCrossTalkSpecification::SerializeJson(const Json::Value &json){

  std::cerr << json.toStyledString();
}


void TraceCrossTalkSpecification::LoadJson(Json::Value & json, const std::string& filename_json)
{
  std::ifstream inJsonFile(filename_json.c_str(), std::ios::in);
  if (inJsonFile.good())
    inJsonFile >> json;
  else
    std::cerr << "[TraceXtalk] Unable to read JSON file " << filename_json << std::endl;
  inJsonFile.close();
}

void TraceCrossTalkSpecification::WriteJson(const Json::Value & json, const std::string& filename_json)
{
  std::ofstream outJsonFile(filename_json.c_str(), std::ios::out);
  if (outJsonFile.good())
    outJsonFile << json.toStyledString();
  else
    std::cerr << "[TraceXtalk] Unable to write JSON file " << filename_json << std::endl;
  outJsonFile.close();
}
*/

void TraceCrossTalkSpecification::LoadJson(Json::Value & json, const std::string& fname)
{
  std::ifstream inJsonFile(fname.c_str(), std::ios::in);
  if (inJsonFile.good())
    inJsonFile >> json;
  else
    std::cerr << "[TraceXtalk] Unable to read JSON file " << fname << std::endl;
  inJsonFile.close();
}

void TraceCrossTalkSpecification::UnpackTraceXtalkInfo(Json::Value &json)
{
  // load chip_proprieties
  chipType_loaded = json["chip_proprieties"]["chip_type"].asString(); // this is not used in any way

  std::string grid_type = json["chip_proprieties"]["grid_type"].asString();
  if (grid_type == "HexPackedColumnsOffset"){
    three_series = false;
    hex_packed = true;
  }
  if (grid_type == "HexPackedRowsOffset"){
    three_series = true;
    hex_packed = true;
  }
  if (grid_type == "SquareGrid"){
    three_series = true;
    hex_packed = false;
  }

  initial_phase = json["chip_proprieties"]["initial_phase"].asInt();
  
  int max_x = json["chip_proprieties"]["max_x"].asInt();
  int max_y = json["chip_proprieties"]["max_y"].asInt();
  if (if_block_analysis){
    max_x = json["chip_proprieties"]["max_x_composite"].asInt();
    max_y = json["chip_proprieties"]["max_y_composite"].asInt();
  }

  // load cross_talk_model
  // These two possibly should be collapsed into "SimplePerBlock"
  // Assumed json["parameters"]=="PerBlock"
  std::string model_type = json["cross_talk_model"]["model"].asString();
  if ( model_type == "Simple"){
    simple_model = true;
  } else { // "Complex"
    simple_model = false;
  }
  
  // do we need to add it to json? as optional?
  rescale_flag = false;
  //rescale_flag = json["cross_talk_model"]["rescale"].asBool();

  int num_neighbor =  json["cross_talk_model"]["number_of_neighbors"].asInt();
  Allocate(num_neighbor);
 
  int num_blocks_x =  json["cross_talk_model"]["number_of_blocks_x"].asInt(); 
  int num_blocks_y =  json["cross_talk_model"]["number_of_blocks_y"].asInt(); 

  int block_x = 0;
  int block_y = 0;
  // this check does not make much sense, paranoia
  if (max_x * max_y * num_blocks_x * num_blocks_y !=0){
    block_x = full_chip_x / (max_x / num_blocks_x);
    block_y = full_chip_y / (max_y / num_blocks_y);
  }

  // kudos for the style: assumed fixed order of blocks, these two lists are not used (still useful in R)
  // json["cross_talk_model"]["blocks_x"] and json["cross_talk_model"]["blocks_y"]
  int block_id = block_y + block_x * num_blocks_y;

  // per neighbour paramters
  for (int ni=0; ni<num_neighbor; ni++){ 
    std::ostringstream neighbour_id;
    neighbour_id << "neighbour_" << std::setfill('0') << std::setw(3) << ni;

    cx[ni] = json["cross_talk_model"][neighbour_id.str()]["offset_x"].asInt();
    cy[ni] = json["cross_talk_model"][neighbour_id.str()]["offset_y"].asInt();

    multiplier[ni] = json["cross_talk_model"][neighbour_id.str()]["multipliers"][block_id].asDouble();

    tau_top[ni] = 1.0f; 
    tau_fluid[ni] = 1.0f;
    if (!simple_model){
      tau_top[ni] = json["cross_talk_model"][neighbour_id.str()]["tau_top"][block_id].asDouble();
      tau_fluid[ni] = json["cross_talk_model"][neighbour_id.str()]["tau_fluid"][block_id].asDouble();
    }
  }
}

void TraceCrossTalkSpecification::PackTraceXtalkInfo(Json::Value &json)
{
  // global paramters
  json["chip_proprieties"]["chip_type"] = chipType_loaded; // this is not used in any way
  
  json["chip_proprieties"]["grid_type"] = "debug me";
  if ( !three_series && hex_packed )
    json["chip_proprieties"]["grid_type"] = "HexPackedColumnsOffset";
  if ( three_series && hex_packed )
    json["chip_proprieties"]["grid_type"] = "HexPackedRowsOffset";
  if ( three_series && !hex_packed )
    json["chip_proprieties"]["grid_type"] = "SquareGrid";

  json["chip_proprieties"]["initial_phase"] = initial_phase;

  // absolute chip coordinates for this region
  json["region_location"]["actual_chip_type"] = chipType; 
  json["region_location"]["block_analysis"] = if_block_analysis;
  json["region_location"]["x"] = full_chip_x;
  json["region_location"]["y"] = full_chip_y;

  // save in a single-block format with parameters from this region
  if (simple_model)
    json["cross_talk_model"]["model"] = "Simple"; 
  else
    json["cross_talk_model"]["model"] = "Complex";

  json["cross_talk_model"]["parameters"] = "SingleRegion"; // same as PerBlock with a single bock
  json["cross_talk_model"]["number_of_neighbors"] = nei_affected;
  json["cross_talk_model"]["number_of_blocks_X"] = 1;
  json["cross_talk_model"]["number_of_blocks_Y"] = 1;
  json["cross_talk_model"]["blocks_x"][0] = 0;
  json["cross_talk_model"]["blocks_y"][0] = 0;

  // per neighbour paramters
  for (int ni=0; ni<nei_affected; ni++){ 
    std::ostringstream neighbour_id;
    neighbour_id << "neighbour_" << std::setfill('0') << std::setw(3) << ni;
    json["cross_talk_model"][neighbour_id.str()]["offset_x"] = cx[ni];
    json["cross_talk_model"][neighbour_id.str()]["offset_y"] = cy[ni];
    json["cross_talk_model"][neighbour_id.str()]["multipliers"][0] = multiplier[ni];
    if (!simple_model){
      json["cross_talk_model"][neighbour_id.str()]["tau_top"][0] = tau_top[ni];
      json["cross_talk_model"][neighbour_id.str()]["tau_fluid"][0] = tau_fluid[ni];
    }
  }  
}

void TraceCrossTalkSpecification::SerializeJson(const Json::Value &json)
{
  std::cerr << json.toStyledString();
}

void TraceCrossTalkSpecification::TestWrite(){
  Json::Value out_json;
  PackTraceXtalkInfo( out_json );
  SerializeJson( out_json );
}

void TraceCrossTalkSpecification::ReadCrossTalkFromFile( std::string &fname )
{
  std::cout << "Loading crosstalk for " << chipType << " from: " << fname << " at (" << full_chip_x << "," << full_chip_y << ")" << std::endl;
  Json::Value in_json;
  LoadJson( in_json, fname );
  UnpackTraceXtalkInfo( in_json );
  // echo loaded & computed parameters for each region, not practical unless debugging
  // TestWrite(); 
}

void TraceCrossTalkSpecification::SetNewQuadGrid()
{
    //printf("Quad Grid set\n");
    hex_packed = false;
    //Allocate(8); // how many neighbors are significant
    Allocate(5); // after commenting out zero neighbors
    // left/right
    int ndx = 0;
    cx[ndx] = -1;
    cy[ndx] = 0;
    tau_top[ndx] = 3;
    tau_fluid[ndx] = 7;
    multiplier[ndx] = 0.02;
    ndx++;// reduced because downstream
    cx[ndx] = 1,  cy[ndx] = 0;
    tau_top[ndx] = 3;
    tau_fluid[ndx] = 7;
    multiplier[ndx] = 0.03;
    ndx++;
    // phase of hex grid shifts these two entities
    // up
    //cx[ndx] = -1; cy[ndx] = 1;  tau_top[ndx] = 3; tau_fluid[ndx] = 7; multiplier[ndx] = 0.00; ndx++;
    cx[ndx] = 0;
    cy[ndx] = 1;
    tau_top[ndx] = 3;
    tau_fluid[ndx] = 7;
    multiplier[ndx] = 0.03;
    ndx++;
    cx[ndx] = 1;
    cy[ndx] = 1;
    tau_top[ndx] = 3;
    tau_fluid[ndx] = 7;
    multiplier[ndx] = 0.02;
    ndx++; // upstream towards inlet neighbor, reduced because diagonal
    // down
    //cx[ndx] = -1; cy[ndx] = -1;  tau_top[ndx] = 3; tau_fluid[ndx] = 7; multiplier[ndx] = 0.00; ndx++;
    cx[ndx] = 0;
    cy[ndx] = -1;
    tau_top[ndx] = 3;
    tau_fluid[ndx] = 7;
    multiplier[ndx] = 0.02;
    ndx++; // reduced because downstream
    //cx[ndx] = 1; cy[ndx] = -1; tau_top[ndx] = 3; tau_fluid[ndx] = 7; multiplier[ndx] = 0.00; ndx++;

}

void TraceCrossTalkSpecification::SetNewHexGrid()
{
    hex_packed = true;
    Allocate(6);
    int ndx = 0;
    // left/right
    cx[ndx] = -1;
    cy[ndx] = 0;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 15;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1; cy[ndx] = 0;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    // phase of hex grid shifts these two entities
    // up
    cx[ndx] = 0;
    cy[ndx] = 1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1;
    cy[ndx] = 1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    // down
    cx[ndx] = 0;
    cy[ndx] = -1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1;
    cy[ndx] = -1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 11;
    multiplier[ndx] = 0.01;
    ndx++;
}

void TraceCrossTalkSpecification::SetNewHexGridP0()
{
    hex_packed = true;
    three_series = false;
    initial_phase = 1;
    Allocate(6);
    int ndx = 0;
    // left/right
    cx[ndx] = 0;    cy[ndx] = -1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1; cy[ndx] = 0;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    // phase of hex grid shifts these two entities
    // up
    cx[ndx] = 0;    cy[ndx] = 1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1;    cy[ndx] = 1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    // down
    cx[ndx] = 0;    cy[ndx] = -1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1;    cy[ndx] = -1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
}

// same spec currently 
void TraceCrossTalkSpecification::SetAggressiveHexGrid()
{
    hex_packed = true;
    three_series = false;
    Allocate(6);
    int ndx = 0;
    // left/right
    cx[ndx] = -1;
    cy[ndx] = 0;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 15;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1, cy[ndx] = 0;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    // phase of hex grid shifts these two entities
    // up
    cx[ndx] = 0;
    cy[ndx] = 1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1;
    cy[ndx] = 1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    // down
    cx[ndx] = 0;
    cy[ndx] = -1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 12;
    multiplier[ndx] = 0.01;
    ndx++;
    cx[ndx] = 1;
    cy[ndx] = -1;
    tau_top[ndx] = 5;
    tau_fluid[ndx] = 11;
    multiplier[ndx] = 0.01;
    ndx++;
}

/*
// changed to json input format
#define MAX_LINE_LEN    512
void TraceCrossTalkSpecification::ReadCrossTalkFromFile(const char *fname)
{
    struct stat fstatus;
    int         status;
    FILE *param_file;
    char *line;
    int nChar = MAX_LINE_LEN;

    int hex_tmp;
    int three_tmp;
    int ndx = 0;
    int num_neighbor=0;
    int ax, ay;
    float tA,tB,tM;

    line = new char[MAX_LINE_LEN];

    status = stat(fname,&fstatus);

    if (status == 0)
    {
        // file exists
        printf("XTALK: loading parameters from %s\n",fname);

        param_file=fopen(fname,"rt");

        bool done = false;

        while (!done)
        {
            int bytes_read = getline(&line,(size_t *)&nChar,param_file);

            line[bytes_read]='\0';
            if (bytes_read > 0)
            {
                if (strncmp("hex_packed",line,10)==0)
                {
                    sscanf(line,"hex_packed: %d", &hex_tmp);
                    if (hex_tmp>0)
                        hex_packed = true;
                    else
                        hex_packed= false;
                    //printf("hex_packed: %d\n", hex_tmp);
                }
                if (strncmp("three_series",line,10)==0)
                {
                    sscanf(line,"three_series: %d", &three_tmp);
                    if (three_tmp>0)
                        three_series = true;
                    else
                        three_series= false;
                    //printf("three_series: %d\n", three_tmp);
                }
                if (strncmp("simple_model",line,12)==0)
                {
                  int simple_tmp;
                    sscanf(line,"simple_model: %d", &simple_tmp);
                    if (simple_tmp>0)
                        simple_model = true;
                    else
                        simple_model= false;
                    //printf("simple_model: %d\n", simple_model);
                }                
                if (strncmp("initial_phase",line,13)==0)
                {
                    sscanf(line,"initial_phase: %d", &initial_phase);
                    //printf("initial_phase: %d\n", initialPhase);
                }
               if (strncmp("rescale_flag",line,12)==0)
                {
                  int rescale_tmp;
                    sscanf(line,"rescale_flag: %d", &rescale_tmp);
                    if (rescale_tmp>0)
                        rescale_flag = true;
                    else
                        rescale_flag= false;
                    //printf("three_series: %d\n", three_tmp);
                }         
                if (strncmp("num_neighbor",line,12)==0)
                {
                    sscanf(line, "num_neighbor: %d", &num_neighbor);
                    Allocate(num_neighbor);
                    ndx = 0;
                    //printf("num_neighbor: %d\n",num_neighbor);
                }

                if (strncmp("a_neighbor",line,10)==0)
                {
                    sscanf(line,"a_neighbor: %d %d %f %f %f", &ax, &ay, &tA, &tB, &tM);
                    //printf("a_neighbor: %d %d %f %f %f\n",ax, ay,tA,tB,tM);
                    cx[ndx] = ax;
                    cy[ndx] = ay;
                    tau_top[ndx] = tA;
                    tau_fluid[ndx] = tB;
                    multiplier[ndx] = tM;
                    ndx++;
                }
            }
            else
                done = true;
        }

        fclose(param_file);

    }
    else
        printf("XTALK: parameter file %s does not exist\n",fname);

    delete [] line;
}
*/

void TraceCrossTalkSpecification::NeighborByGridPhase(int &ncx, int &ncy, int cx, int cy, int cxd, int cyd, int phase)
{
    if (phase==0)
    {
        ncx = cx+cxd;
        ncy = cy+cyd;
    } else
    {
        ncy = cy+cyd;
        if (cyd!=0)
            ncx = cx+cxd-phase; // up/down levels are offset alternately on rows
        else
            ncx = cx+cxd;
    }
    // unless we're in a hex grid and need to know our offset before deciding
    // however those variables need to be passed
}

// bb operates the other direction

void TraceCrossTalkSpecification::NeighborByGridPhaseBB(int &ncx, int &ncy, int cx, int cy, int cxd, int cyd, int phase)
{
        ncx = cx+cxd; // neighbor columns are always correct
        ncy = cy+cyd; // map is correct
        if ((phase!=0) & (((cxd+16) %2)!=0)) //neighbors may be more than one away!!!!
            ncy -= phase; // up/down levels are offset alternately on cols
}

void TraceCrossTalkSpecification::NeighborByChipType(int &ncx, int &ncy, int bead_rx, int bead_ry, int nei_idx, int region_x, int region_y)
{
  // the logic has now become complex, so encapsulate it
        // phase for hex-packed
      if (!hex_packed)
        NeighborByGridPhase (ncx,ncy,bead_rx,bead_ry,cx[nei_idx],cy[nei_idx], 0);
      else
      {
        if (three_series)
          NeighborByGridPhase (ncx,ncy,bead_rx,bead_ry,cx[nei_idx],cy[nei_idx], (region_y + bead_ry+1) % 2); // maybe????
        else
          NeighborByGridPhaseBB(ncx,ncy,bead_rx,bead_ry,cx[nei_idx],cy[nei_idx], (region_x + bead_rx+1+initial_phase) % 2); // maybe????
      }

}



void TraceCrossTalkSpecification::BootUpXtalkSpec( bool can_do_xtalk_correction, std::string &fname, 
						    std::string &_chipType, bool _if_block_analysis, 
						    int _full_chip_x, int _full_chip_y )
{
  chipType = _chipType;
  full_chip_x = _full_chip_x;
  full_chip_y = _full_chip_y;
  if_block_analysis = _if_block_analysis;

  if (can_do_xtalk_correction)
    {
      if (fname.length()>0) 
        {
	  ReadCrossTalkFromFile( fname );
        } else {
            if ( chipType == "318" || chipType == "316v2" )
                SetNewHexGrid(); // find out who we really are!

        else if( chipType == "p1.1.17" || chipType == "540" || chipType == "541" || chipType == "p2.1.1" || chipType == "p2.3.1" || chipType == "p1.1.541" )
                SetAggressiveHexGrid(); // 900 may have different cross-talk!

	    else if( chipType == "p1.0.19") 
                SetNewHexGridP0();

	    // load  default json files for P-zero series
	    else if( chipType == "p1.0.20" 
		   ||  chipType == "530" 
		   ||  chipType == "520" 
           ||  chipType == "531"
           ||  chipType == "521"
           ||  chipType == "522"
           ||  chipType == "p2.0.1")
        {

		string filename = "xtalk.trace.";
		filename += chipType;
		filename += ".json";

		char *tmp_config_file = NULL;
		tmp_config_file = GetIonConfigFile (filename.c_str());
 
		// an ugly construction to be removed once all other hard-codded xtalk paramters are removed
		if(tmp_config_file == NULL)
		  {
		    SetNewHexGridP0();
		  }
		else
		  {
		    filename = tmp_config_file;
		    free(tmp_config_file);
		    ReadCrossTalkFromFile( filename  );
		  }
		  
	      }

            else
                SetNewQuadGrid();
        }
    }
}
