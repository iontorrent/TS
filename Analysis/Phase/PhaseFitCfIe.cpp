/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PhaseFitCfIe.h"


/*
** Fit one CF and one IE value to all flows
*/

void PhaseFitCfIe::Evaluate(float *pred, float *param) {
  CfIeParam *p = (CfIeParam *)param;

  weight_vec_t cfTry(1,p->cf);
  weight_vec_t ieTry(1,p->ie);

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, dr, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, dr, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
  }

  // Advancers are in place, now do the prediction
  unsigned int nFlowPerCycle = flowCycle.size();
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}


/*
** Fit one CF, one IE and one DR value to all flows
*/

void PhaseFitCfIeDr::Evaluate(float *pred, float *param) {
  CfIeDrParam *p = (CfIeDrParam *)param;

  weight_vec_t cfTry(1,p->cf);
  weight_vec_t ieTry(1,p->ie);
  weight_vec_t drTry(1,p->dr);

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, drTry, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, drTry, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
  }

  // Advancers are in place, now do the prediction
  unsigned int nFlowPerCycle = flowCycle.size();
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}


/*
** Fit one CF, one IE, one DR and one hpScale value to all flows
*/

void PhaseFitCfIeDrHpScale::Evaluate(float *pred, float *param) {
  CfIeDrHpScaleParam *p = (CfIeDrHpScaleParam *)param;

  weight_vec_t cfTry(1,p->cf);
  weight_vec_t ieTry(1,p->ie);
  weight_vec_t drTry(1,p->dr);
  weight_vec_t hpScaleTry(1,p->hpScale);

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, drTry, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, drTry, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
    read[iRead].setHpScale(hpScaleTry);
  }

  // Advancers are in place, now do the prediction
  unsigned int nFlowPerCycle = flowCycle.size();
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}


/*
** Fit one hpScale value to all flows
*/

void PhaseFitHpScale::Evaluate(float *pred, float *param) {
  HpScaleParam *p = (HpScaleParam *)param;

  weight_vec_t hpScaleTry(1,p->hpScale);

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentration, cf, ie, dr, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentration, cf, ie, dr, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
    read[iRead].setHpScale(hpScaleTry);
  }

  // Advancers are in place, now do the prediction
  unsigned int nFlowPerCycle = flowCycle.size();
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}


/*
** Fit one CF, one IE, one DR and four hpScale values to all flows
*/

void PhaseFitCfIeDrHpScale4::Evaluate(float *pred, float *param) {
  CfIeDrHpScale4Param *p = (CfIeDrHpScale4Param *)param;

  weight_vec_t cfTry(1,p->cf);
  weight_vec_t ieTry(1,p->ie);
  weight_vec_t drTry(1,p->dr);
  weight_vec_t hpScaleTry(0);
  hpScaleTry.push_back(p->hpScaleA);
  hpScaleTry.push_back(p->hpScaleC);
  hpScaleTry.push_back(p->hpScaleG);
  hpScaleTry.push_back(p->hpScaleT);

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, drTry, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, drTry, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
    read[iRead].setHpScale(hpScaleTry);
  }

  // Advancers are in place, now do the prediction
  unsigned int nFlowPerCycle = flowCycle.size();
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}


/*
** Fit four hpScale values to all flows
*/

void PhaseFitHpScale4::Evaluate(float *pred, float *param) {
  HpScale4Param *p = (HpScale4Param *)param;

  weight_vec_t hpScaleTry(0);
  hpScaleTry.push_back(p->hpScaleA);
  hpScaleTry.push_back(p->hpScaleC);
  hpScaleTry.push_back(p->hpScaleG);
  hpScaleTry.push_back(p->hpScaleT);

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentration, cf, ie, dr, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentration, cf, ie, dr, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
    read[iRead].setHpScale(hpScaleTry);
  }

  // Advancers are in place, now do the prediction
  unsigned int nFlowPerCycle = flowCycle.size();
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}


/*
** Fit the 12 off-diagonal components of a nuc contamination matrix
*/

void PhaseFitNucContam::Evaluate(float *pred, float *param) {
  NucContamParam *p = (NucContamParam *)param;

  vector<weight_vec_t> concentrationTry(N_NUCLEOTIDES);
  for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++) {
    concentrationTry[iNuc].resize(N_NUCLEOTIDES);
    concentrationTry[iNuc][iNuc] = 1;
  }
  concentrationTry[0][1] = p->C_in_A;
  concentrationTry[0][2] = p->G_in_A;
  concentrationTry[0][3] = p->T_in_A;
  concentrationTry[1][0] = p->A_in_C;
  concentrationTry[1][2] = p->G_in_C;
  concentrationTry[1][3] = p->T_in_C;
  concentrationTry[2][0] = p->A_in_G;
  concentrationTry[2][1] = p->C_in_G;
  concentrationTry[2][3] = p->T_in_G;
  concentrationTry[3][0] = p->A_in_T;
  concentrationTry[3][1] = p->C_in_T;
  concentrationTry[3][2] = p->G_in_T;

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentrationTry, cf, ie, dr, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentrationTry, cf, ie, dr, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
  }

  // Advancers are in place, now do the prediction
  unsigned int nFlowPerCycle = flowCycle.size();
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}


/*
** Fit one CF and 4 nuc-specific IE values
*/

void PhaseFitCfIe4::Evaluate(float *pred, float *param) {
  CfIe4Param *p = (CfIe4Param *)param;

  weight_vec_t cfTry(1,p->cf);

  // remap per-nuc ie values to a vector of length flowCycle
  weight_vec_t ieByNuc(N_NUCLEOTIDES);
  ieByNuc[0] = p->ieA;
  ieByNuc[1] = p->ieC;
  ieByNuc[2] = p->ieG;
  ieByNuc[3] = p->ieT;
  unsigned int nFlowPerCycle = flowCycle.size();
  weight_vec_t ieTry(nFlowPerCycle);
  for(unsigned int iFlow=0; iFlow<nFlowPerCycle; iFlow++)
    ieTry[iFlow] = ieByNuc[flowCycle[iFlow]];

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, dr, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentration, cfTry, ieTry, dr, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
  }

  // Advancers are in place, now do the prediction
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}



/*
** Fit ie and the 12 off-diagonal components of a nuc contamination matrix
*/

void PhaseFitNucContamIe::Evaluate(float *pred, float *param) {
  NucContamIeParam *p = (NucContamIeParam *)param;

  weight_vec_t ieTry(1,p->ie);

  vector<weight_vec_t> concentrationTry(N_NUCLEOTIDES);
  for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; iNuc++) {
    concentrationTry[iNuc].resize(N_NUCLEOTIDES);
    concentrationTry[iNuc][iNuc] = 1;
  }
  concentrationTry[0][1] = p->C_in_A;
  concentrationTry[0][2] = p->G_in_A;
  concentrationTry[0][3] = p->T_in_A;
  concentrationTry[1][0] = p->A_in_C;
  concentrationTry[1][2] = p->G_in_C;
  concentrationTry[1][3] = p->T_in_C;
  concentrationTry[2][0] = p->A_in_G;
  concentrationTry[2][1] = p->C_in_G;
  concentrationTry[2][3] = p->T_in_G;
  concentrationTry[3][0] = p->A_in_T;
  concentrationTry[3][1] = p->C_in_T;
  concentrationTry[3][2] = p->G_in_T;

  // Prepare all reads for prediction
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    if(iRead==0) {
      // We only need to compute advancers for one read, they'll be the same for the rest.
      read[iRead].setAdvancerWeights(concentrationTry, cf, ieTry, dr, extendAdvancerFirst, droopAdvancerFirst, true );
      read[iRead].setAdvancerWeights(concentrationTry, cf, ieTry, dr, extendAdvancer,      droopAdvancer,      false);
    }
    read[iRead].resetTemplate();
  }

  // Advancers are in place, now do the prediction
  unsigned int nFlowPerCycle = flowCycle.size();
  unsigned int iPred=0;
  for(unsigned int iRead=0; iRead<read.size(); iRead++) {
    bool firstCycle = true;
    for(unsigned int iFlow=0; iFlow<nFlow; iFlow++, iPred++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      weight_t thisSig;
      if(firstCycle)
        thisSig = read[iRead].applyFlow(iFlow, extendAdvancerFirst, droopAdvancerFirst);
      else
        thisSig = read[iRead].applyFlow(iFlow % flowCycle.size(), extendAdvancer, droopAdvancer);
      pred[iPred] = (float) thisSig;
    }
  }
}
