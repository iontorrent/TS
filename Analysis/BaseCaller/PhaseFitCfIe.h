/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PHASEFITCFIE_H
#define PHASEFITCFIE_H

#include "PhaseFit.h"
#include "LevMarFitter.h"

using namespace std;


/*
** Fit one CF and one IE value to all flows
*/

struct CfIeParam {
  float cf;
  float ie;
};

class PhaseFitCfIe : public PhaseFit
{
public:
  void SetParamMax(CfIeParam p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(CfIeParam p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(CfIeParam p) { params = p; };
  CfIeParam GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(CfIeParam)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  CfIeParam params;
  CfIeParam min_params, max_params;
};



/*
** Fit one CF, one IE and one DR value to all flows
*/

struct CfIeDrParam {
  float cf;
  float ie;
  float dr;
};

class PhaseFitCfIeDr : public PhaseFit
{
public:
  void SetParamMax(CfIeDrParam p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(CfIeDrParam p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(CfIeDrParam p) { params = p; };
  CfIeDrParam GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(CfIeDrParam)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  CfIeDrParam params;
  CfIeDrParam min_params, max_params;
};



/*
** Fit one CF, one IE, one DR and one hpScale value to all flows
*/

struct CfIeDrHpScaleParam {
  float cf;
  float ie;
  float dr;
  float hpScale;
};

class PhaseFitCfIeDrHpScale : public PhaseFit
{
public:
  void SetParamMax(CfIeDrHpScaleParam p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(CfIeDrHpScaleParam p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(CfIeDrHpScaleParam p) { params = p; };
  CfIeDrHpScaleParam GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(CfIeDrHpScaleParam)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  CfIeDrHpScaleParam params;
  CfIeDrHpScaleParam min_params, max_params;
};



/*
** Fit one hpScale value to all flows
*/

struct HpScaleParam {
  float hpScale;
};

class PhaseFitHpScale : public PhaseFit
{
public:
  void SetParamMax(HpScaleParam p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(HpScaleParam p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(HpScaleParam p) { params = p; };
  HpScaleParam GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(HpScaleParam)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  HpScaleParam params;
  HpScaleParam min_params, max_params;
};



/*
** Fit one CF, one IE, one DR and four hpScale values to all flows
*/

struct CfIeDrHpScale4Param {
  float cf;
  float ie;
  float dr;
  float hpScaleA;
  float hpScaleC;
  float hpScaleG;
  float hpScaleT;
};

class PhaseFitCfIeDrHpScale4 : public PhaseFit
{
public:
  void SetParamMax(CfIeDrHpScale4Param p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(CfIeDrHpScale4Param p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(CfIeDrHpScale4Param p) { params = p; };
  CfIeDrHpScale4Param GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(CfIeDrHpScale4Param)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  CfIeDrHpScale4Param params;
  CfIeDrHpScale4Param min_params, max_params;
};



/*
** Fit four hpScale values to all flows
*/

struct HpScale4Param {
  float hpScaleA;
  float hpScaleC;
  float hpScaleG;
  float hpScaleT;
};

class PhaseFitHpScale4 : public PhaseFit
{
public:
  void SetParamMax(HpScale4Param p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(HpScale4Param p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(HpScale4Param p) { params = p; };
  HpScale4Param GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(HpScale4Param)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  HpScale4Param params;
  HpScale4Param min_params, max_params;
};



/*
** Fit the 12 off-diagonal components of a nuc contamination matrix
*/

struct NucContamParam {
                float C_in_A; float G_in_A; float T_in_A;
  float A_in_C;               float G_in_C; float T_in_C;
  float A_in_G; float C_in_G;               float T_in_G;
  float A_in_T; float C_in_T;               float G_in_T;
};

class PhaseFitNucContam : public PhaseFit
{
public:
  void SetParamMax(NucContamParam p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(NucContamParam p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(NucContamParam p) { params = p; };
  NucContamParam GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(NucContamParam)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  NucContamParam params;
  NucContamParam min_params, max_params;
};



/*
** Fit one CF and 4 nuc-specific IE values
*/

struct CfIe4Param {
  float cf;
  float ieA;
  float ieC;
  float ieG;
  float ieT;
};

class PhaseFitCfIe4 : public PhaseFit
{
public:
  void SetParamMax(CfIe4Param p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(CfIe4Param p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(CfIe4Param p) { params = p; };
  CfIe4Param GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(CfIe4Param)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  CfIe4Param params;
  CfIe4Param min_params, max_params;
};




/*
** Fit ie and the 12 off-diagonal components of a nuc contamination matrix
*/

struct NucContamIeParam {
  float ie;
                float C_in_A; float G_in_A; float T_in_A;
  float A_in_C;               float G_in_C; float T_in_C;
  float A_in_G; float C_in_G;               float T_in_G;
  float A_in_T; float C_in_T;               float G_in_T;
};

class PhaseFitNucContamIe : public PhaseFit
{
public:
  void SetParamMax(NucContamIeParam p) { max_params = p; LevMarFitter::SetParamMax((float *) &max_params); };
  void SetParamMin(NucContamIeParam p) { min_params = p; LevMarFitter::SetParamMin((float *) &min_params); };
  void SetParam(NucContamIeParam p) { params = p; };
  NucContamIeParam GetParam(void) { return(params); };
  int LevMarFit(int max_iter) { int nParam = sizeof(NucContamIeParam)/sizeof(float); return(PhaseFit::LevMarFit(max_iter, nParam, (float *) &params)); };
  virtual void Evaluate(float *pred, float *param);
private:
  NucContamIeParam params;
  NucContamIeParam min_params, max_params;
};

#endif // PHASEFITCFIE_H
