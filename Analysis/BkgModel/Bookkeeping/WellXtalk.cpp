/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */
#include "WellXtalk.h"

// default coefficients for PI
// using complex methodology
float nn_even_phase_map_defaultPI[] = {
  0.000,0.000,0.010,0.000,0.000,
  0.015,0.031,0.131,0.026,0.015,
  0.026,0.127,0.000,0.111,0.025,
  0.015,0.117,0.133,0.110,0.013,
  0.000,0.021,0.010,0.020,0.000,
};


WellXtalk::WellXtalk(){
  nn_span_x = nn_span_y =0;
  map_span = 0;
 simple_xtalk = true;
 grid_type = 0;
 chip_type = "none_set";
}

// even assumed default
// entries outside the matrix assumed zero
// columns assumed shifted
void WellXtalk::OddFromEvenHexCol(){
  int double_y = 2*nn_span_y+1;
  int double_x = 2*nn_span_x+1;
  for (int lr=0; lr<double_y; lr++){
    for (int lc=0; lc<double_x; lc++){
      int  tc = lc;
      int tr = lr;
      int col_phase = lc &1;
      if (col_phase)
        tr = lr+1;
      if (tr<double_y){
        nn_odd_phase_map[Hash(lr,lc)] = nn_even_phase_map[Hash(tr,tc)];
      } else {
        nn_odd_phase_map[Hash(lr,lc)] = 0.0f;
      }
    }
  }
}

void WellXtalk::OddFromEven(){
  // no transformation needed, just copy the coefficients
  nn_odd_phase_map = nn_even_phase_map;
}

// 0-2*nn_span_x
int WellXtalk::Hash(int lr, int lc){
  return(lr*(2*nn_span_x+1)+lc);
}

void WellXtalk::Allocate(int x_span, int y_span){
  nn_span_x = x_span;
  nn_span_y = y_span;
  map_span = (2*nn_span_x+1)*(2*nn_span_y+1);
  nn_odd_phase_map.assign(map_span,0);
  nn_even_phase_map.assign(map_span,0);
}

void WellXtalk::DefaultPI(){
  Allocate(2,2);
  simple_xtalk = false;
  chip_type = "P1.1.17";
  if (map_span==25){
    memcpy(&(nn_even_phase_map[0]),nn_even_phase_map_defaultPI,sizeof(float[25]));
    OddFromEvenHexCol();
    NormalMap();
  }
}

void WellXtalk::DefaultZeroXTalk(){
  Allocate(2,2);
  // zero coefficients, yet bead corrector still nonzero
}

void WellXtalk::NormalMap(){
  NormalizeCoeffs(&(nn_even_phase_map[0]));
  NormalizeCoeffs(&(nn_odd_phase_map[0]));
}

// note: does >not< handle the case when all coefficients are zero correctly
// legacy of complex distortion method
void WellXtalk::NormalizeCoeffs(float *c_map){
  float coeff_sum = 0.0f;
  for (int i=0; i<map_span; i++)
    coeff_sum += c_map[i];
  for (int i=0; i<map_span; i++){
    c_map[i] /= coeff_sum;
  }
}

WellXtalk::~WellXtalk()
{

}


float WellXtalk::UnweaveMap(float *ampl_map, int row, int col, Region *region, float default_signal, int phase)
{
  float sum = 0.0f;
  float *coeffs;
  //float coeff_sum = 0.0f;
  float coeff;

  int lr,lc;
  if (phase)
    coeffs = &(nn_even_phase_map[0]);
  else
    coeffs = &(nn_odd_phase_map[0]);

  for (int r=row-nn_span_y;r<=(row+nn_span_y);r++)
  {
    lr = r-row+nn_span_y;
    for (int c=col-nn_span_x;c<=(col+nn_span_x);c++)
    {
      lc = c-col+nn_span_x;
      coeff = coeffs[Hash(lr,lc)];

      if ((r < 0) || (r>=region->h) || (c < 0) || (c>=region->w))
      {
        // if we are on the edge of the region...as a stand-in for actual data
        // use the region average signal
        sum += default_signal*coeff;
      }
      else
      {
        sum += ampl_map[r*region->w+c]*coeff;
      }

      //coeff_sum += coeff;
    }
  }

  //sum /= coeff_sum;
  return(sum);
}

// i/o from environment

void WellXtalk::LoadJson(Json::Value & json, const std::string& filename_json)
{
  std::ifstream inJsonFile(filename_json.c_str(), std::ios::in);
  if (inJsonFile.good())
    inJsonFile >> json;
  else
    std::cerr << "[WellXtalk] Unable to read JSON file " << filename_json << std::endl;
  inJsonFile.close();
}

void WellXtalk::WriteJson(const Json::Value & json, const std::string& filename_json)
{
  std::ofstream outJsonFile(filename_json.c_str(), std::ios::out);
  if (outJsonFile.good())
    outJsonFile << json.toStyledString();
  else
    std::cerr << "[WellXtalk] Unable to write JSON file " << filename_json << std::endl;
  outJsonFile.close();
}

void WellXtalk::SerializeJson(const Json::Value &json){

  std::cerr << json.toStyledString();
}

void WellXtalk::PackWellXtalkInfo(Json::Value &json)
{
  // this is the chip_type in the cross-talk matrix, not necessarily the chip_type it is applied to
  json["ChipType"] = chip_type; // currently only valid, used to check utility
  json["MapSpan"] =map_span;
  for (int mi=0; mi<map_span; mi++){
    json["XtalkCoef"][mi] =nn_even_phase_map[mi];
  }
  json["NNX"] =nn_span_x;
  json["NNY"] =nn_span_y;
  if (grid_type==0)
    json["GridType"] = "SquareGrid";
  if (grid_type==1)
    json["GridType"]  = "HexGridCol";
  if (grid_type==2)
    json["GridType"] = "HexGridRow";
  json["Xtype"] = simple_xtalk ? "Simple" : "Distorted";
}

void WellXtalk::SetGridFromType(){
  if (grid_type==0)
    OddFromEven();
  if (grid_type==1)
    OddFromEvenHexCol();
  if (grid_type==2)
    std::cout << "[WellXtalk] HexGridRow Not implemented yet" << std::endl;
}

void WellXtalk::UnpackWellXtalkInfo(Json::Value &json){
  int nnx = json["NNX"].asInt();
  int nny = json["NNY"].asInt();
 Allocate(nnx,nny);
  for (int mi=0; mi<map_span; mi++){
   nn_even_phase_map[mi] = json["XtalkCoef"][mi].asDouble();
  }
  // assume hex grid col, detect from file
  std::string grid_guess = json["GridType"].asString();
  if (grid_guess.compare("SquareGrid")==0)
    grid_type = 0;
  if (grid_guess.compare("HexGridCol")==0)
    grid_type = 1;
  if (grid_guess.compare("HexGridRow")==0)
    grid_type = 2;

  OddFromEven(); // always something in odd map
  SetGridFromType();
  // this is the chip type specified >in the cross-talk matrix<
  chip_type = json["ChipType"].asString(); // what chip type do we mean?

 // assume simple xtalk, so no normalization
  simple_xtalk = true;
}

void WellXtalk::TestWrite(){
  std::string my_file = "well.xtalk.settings.json";
  Json::Value out_json;
  PackWellXtalkInfo(out_json);
  WriteJson(out_json,my_file);
  SerializeJson(out_json);
}

void WellXtalk::ReadFromFile(std::string &my_file){

  Json::Value in_json;
  LoadJson(in_json, my_file);
  UnpackWellXtalkInfo(in_json);
  TestWrite(); // echo what I read
}

float modulate_effect_by_flow(float start_frac, float flow_num, float offset)
{
  float approach_one_rate = flow_num/(flow_num+offset);
  return ( (1.0f-start_frac) * approach_one_rate + start_frac);
}


TerribleConstants::TerribleConstants(){
  const_frac    = 0.25f;
  const_frac_ref = 0.86f;
  const_frac_ref_complement = 0.14f;
  magic_flow = 32.0f;
  magic_lambda = 1.425f;
  magic_discount = 0.33f;
  cscale    =  1.0f;
  cscale_ref = 1.0f;
  magic_ref_constant =  (const_frac_ref_complement/const_frac_ref) *cscale_ref;
  magic_hplus_ref = 0.0f;
  magic_cscale_constant = magic_discount * cscale;
}

void TerribleConstants::SetupFlow(int flow_num, float region_mean_sig){
  cscale    =  modulate_effect_by_flow(const_frac,     flow_num,magic_flow);
  cscale_ref = modulate_effect_by_flow(const_frac_ref, flow_num, magic_flow);
  magic_ref_constant =  (const_frac_ref_complement/const_frac_ref) *cscale_ref;
  magic_hplus_ref = magic_ref_constant*region_mean_sig;
  magic_cscale_constant = magic_discount * cscale;
}

// "the contribution from the neighbors discounted by the dampening effect of buffering
// "the contribution already accounted for from the reference wells used
// "which sense the mean region signal
// "all on the scale set by the number of copies in this bead
// plus some mysterious values
// the whole effect phased in slowly over flows to maximum

float TerribleConstants::ComputeCorrector(float etbR, float bead_corrector){
  float magic_bead_corrector = (magic_cscale_constant*etbR*bead_corrector);
  float magic_hplus_corrector = magic_lambda* ( magic_bead_corrector -
                                                magic_hplus_ref);
  return(magic_hplus_corrector);
}
