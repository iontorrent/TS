/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

#ifndef WELLXTALK_H
#define WELLXTALK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "json/json.h"
#include "Region.h"
#include "Serialization.h"


class TerribleConstants{
public:
  float const_frac ;
  float const_frac_ref;
  float const_frac_ref_complement;
  float magic_flow;
  float cscale;
  float cscale_ref ;
  float magic_ref_constant;
  float magic_hplus_ref;
  float magic_lambda;
  float magic_discount;
  float magic_cscale_constant;
  TerribleConstants();
  void SetupFlow(int flow_num, float region_mean_sig);
  float ComputeCorrector(float etbR, float bead_corrector);
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
        & const_frac
        & const_frac_ref
        &const_frac_ref_complement
        &magic_flow
        &cscale
        &cscale_ref
        &magic_ref_constant
        &magic_hplus_ref
        &magic_lambda
        &magic_discount
        &magic_cscale_constant;
  }
};

class SimpleCorrector{
public:
  float additive_leakage; // how much mean signal leaks through in a flow i.e. 0.1 = 10%
  float additive_time_scale; // how quickly we approach max leakage - should possibly be spline?
  float additive_leak_offset; // leakage offset - don't start immediately
  float multiplicative_distortion; // mimic 'key' normalization rescaling done by previous additive offset
  SimpleCorrector(){
    additive_leakage = 0.0f;
    additive_time_scale = 32.0f;
    additive_leak_offset = 0.0f;
    multiplicative_distortion = 1.0f;
  };
  void FromJSON(Json::Value &json);
  void ToJSON(Json::Value &json);
  float AdditiveDistortion(int flow_num, float region_mean_sig);
  float MultiplicativeDistortion(int flow_num);
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar
        & additive_leakage
        & additive_time_scale
        & additive_leak_offset
        & multiplicative_distortion;
  }

};


class WellXtalk{
public:
  // hacky cross-talk info
  std::vector<float> nn_odd_phase_map;
  std::vector<float> nn_even_phase_map;

  int nn_span_x;
  int nn_span_y;
  int map_span;

  TerribleConstants my_empirical_function;
  bool simple_xtalk;

  SimpleCorrector my_bkg_distortion;

  // 0=SquareGrid, 1=HexGridCol 2=HexGridRow
  int grid_type;
  std::string chip_type; // what to report as the chip type in the file

  WellXtalk();
  void DefaultPI();
  void DefaultZeroXTalk();
  ~WellXtalk();
  int Hash(int lr, int lc);
  void SetGridFromType();
  void OddFromEvenHexCol();
  void OddFromEven();
  void NormalizeCoeffs(float *c_map);
  void NormalMap();
  void Allocate(int x_span, int y_span);
  float UnweaveMap(float *ampl_map, int row, int col, Region *region, float default_signal, int phase);

  // i/o so we can read cross-talk by chip
  void LoadJson(Json::Value & json, const std::string& filename_json);
  void WriteJson(const Json::Value & json, const std::string& filename_json);
  void SerializeJson(const Json::Value & json);
  void PackWellXtalkInfo(Json::Value &json);
  void UnpackWellXtalkInfo(Json::Value &json);
  void TestWrite();
  void ReadFromFile(std::string &my_file);

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar
        & nn_odd_phase_map
        & nn_even_phase_map
        &nn_span_x
        &nn_span_y
        &map_span
        & my_empirical_function
        & my_bkg_distortion
        & grid_type
        & chip_type
        & simple_xtalk;
  }

};

#endif //WELLXTALK_H
