/* Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved */

//! @file     MultiFlowDist.cpp
//! @ingroup  VariantCaller
//! @brief    HP Indel detection

#include "MultiFlowDist.h"



void MultiFlowDist::SetupAllAlleles(vcf::Variant ** candidate_variant, const string & local_contig_sequence, ExtendParameters *parameters, InputStructures &global_context) {
  multi_variant.SetupAllAlleles(candidate_variant, local_contig_sequence, parameters, global_context);
  multi_variant.FilterAllAlleles(candidate_variant, parameters->my_controls.filter_variant); // flowdist wants filtering here
  AllocateFlowDistVector(candidate_variant, global_context);
}

void MultiFlowDist::AllocateFlowDistVector(vcf::Variant ** candidate_variant, InputStructures &global_context) {
  vector<string> fwdRefObservation = (*candidate_variant)->info["SRF"];
  vector<string> revRefObservation = (*candidate_variant)->info["SRR"];
  vector<string> fwdAltObservation = (*candidate_variant)->info["SAF"];
  vector<string> revAltObservation = (*candidate_variant)->info["SAR"];
  int fwdRef = 0;
  int revRef = 0;
  int fwdDepth = 0;
  int revDepth = 0;
  vector<int> fwdAlt;
  vector<int> revAlt;
  uint8_t totalAlts = (*candidate_variant)->alt.size();

  if (fwdRefObservation.size() > 0)
    fwdRef = atoi(fwdRefObservation.at(0).c_str());

  if (revRefObservation.size() > 0)
    revRef = atoi(revRefObservation.at(0).c_str());

  if (fwdAltObservation.size() == totalAlts && revAltObservation.size() == totalAlts) {
    for (uint8_t i = 0; i < totalAlts; i++) {
      fwdAlt.push_back(atoi(fwdAltObservation.at(i).c_str()));
      revAlt.push_back(atoi(revAltObservation.at(i).c_str()));
    }
  }

  fwdDepth = fwdRef;
  revDepth = revRef;
  for (uint8_t i = 0; i < totalAlts; i++) {
    fwdDepth += fwdAlt.at(i);
    revDepth += revAlt.at(i);
  }

  for (uint8_t i = 0; i < totalAlts; i++) {
    FlowDist * flowDist = new FlowDist((*candidate_variant)->position, multi_variant.seq_context,
        global_context.nFlows, global_context.DEBUG);
    flowDist->summary_stats.setBasePlusDepth(fwdDepth);
    flowDist->summary_stats.setBaseNegDepth(revDepth);
    flowDist->summary_stats.setBasePlusVariant(fwdAlt.at(i));
    flowDist->summary_stats.setBaseNegVariant(revAlt.at(i));
    flowDistVector.push_back(flowDist);

  }
}

void MultiFlowDist::ReceiveStack(StackPlus &my_stack,
                                 InputStructures &global_context,
                                 const string &local_contig_sequence) {
  for (unsigned int i_read=0; i_read<my_stack.read_stack.size(); i_read++) {
    UpdateFlowDistFromEvaluateSoloRead(my_stack.read_stack[i_read],  global_context, local_contig_sequence);
  }
}


// send a single read evaluation to the flowDist structure
void MultiFlowDist::UpdateFlowDistFromEvaluateSoloRead(
  ExtendedReadInfo &current_read,
  InputStructures &global_context,
  const string &local_contig_sequence) {

  int numberAltAlleles = flowDistVector.size();
  vector<float> refLikelihoodVector(numberAltAlleles);
  vector<float> varLikelihoodVector(numberAltAlleles);
  vector<float> minDeltaVector(numberAltAlleles);
  vector<float> minDistToReadVector(numberAltAlleles);
  float maxVarLikelihood = -9999.0;
  int maxFlowDistIndex = -1;

  HypothesisEvaluator  hypEvaluator(current_read.measurementValue,
                                    global_context.treePhaserFlowOrder,
                                    current_read.phase_params,
                                    global_context.DEBUG);



  //loop thru all the alleles and evaluate one by one
  for (size_t counter = 0; counter < flowDistVector.size(); counter++) {
    FlowDist * flowDist = flowDistVector[(int)counter];
    AlleleIdentity variant_identity = multi_variant.allele_identity_vector[counter];

    int variant_start_pos = variant_identity.modified_start_pos;
    int evaluate_splicing, evaluate_success;
    float refLikelihood = 0.0f;
    float minDelta = 9999999.0f;
    float minDistToRead = 999999.0f;
    float varLikelihood = 0.0f;

    if (variant_identity.status.isSNP || variant_identity.status.isMNV || variant_identity.Ordinary()) {
        vector<string> hypotheses;
        evaluate_splicing = HypothesisSpliceVariant(hypotheses, current_read, variant_identity,
                                         (*(multi_variant.variant))->ref, local_contig_sequence, global_context.DEBUG);

        if (evaluate_splicing == 0) {
          evaluate_success = hypEvaluator.calculateCrudeLikelihood(hypotheses, current_read.start_flow, refLikelihood,  minDelta, minDistToRead,  varLikelihood);
          if (evaluate_success == 0) {
            refLikelihoodVector[counter] = refLikelihood;
            varLikelihoodVector[counter] = varLikelihood;
            minDeltaVector[counter] = minDelta;
            minDistToReadVector[counter] = minDistToRead;
            if (varLikelihood > maxVarLikelihood) {
              maxVarLikelihood = varLikelihood;
              maxFlowDistIndex = counter;

            }
          }
        }
    }
    else {
      EvaluateFlowDistForLong(hypEvaluator, current_read, global_context, local_contig_sequence, flowDist, variant_start_pos, variant_identity);

    }
    //EvaluateOneFlowDist(hypEvaluator, current_read, global_context, local_contig_sequence, flowDistVector[(int)counter], multi_variant.variant_identity_vector[counter].modified_start_pos, multi_variant.variant_identity_vector[(int)counter]);
  }//end loop thru alt alleles
  //now find the allele with max likelihood and attribute the read to the best allele.



  if (maxFlowDistIndex >= 0) {
    FlowDist * flowDist = flowDistVector[maxFlowDistIndex];
    AlleleIdentity variant_identity = multi_variant.allele_identity_vector[maxFlowDistIndex];
    float refLikelihood = refLikelihoodVector[maxFlowDistIndex];
    float varLikelihood = varLikelihoodVector[maxFlowDistIndex];
    //float minDelta = minDeltaVector[maxFlowDistIndex];
    float minDistToRead = minDistToReadVector[maxFlowDistIndex];
    int variant_start_pos = variant_identity.modified_start_pos;

    UpdateFlowDistWithOrdinaryVariant(flowDist,current_read.is_forward_strand, refLikelihood, minDistToRead, varLikelihood);

    if (variant_identity.status.isOverCallUnderCallSNP) {
        ExtraEvalForOverUnderSNP(hypEvaluator, current_read, global_context, local_contig_sequence, flowDist, variant_start_pos, variant_identity);
     }
  }
}

void MultiFlowDist::ExtraEvalForOverUnderSNP(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
    InputStructures &global_context,
    const string &local_contig_sequence,
    FlowDist *flowDist, int variant_start_pos,
    AlleleIdentity &variant_identity) {
  uint32_t flowPosition = 0;
  flowPosition = retrieve_flowpos(current_read.alignment.QueryBases, local_contig_sequence, current_read.is_forward_strand, current_read.ref_aln, current_read.seq_aln, current_read.start_pos,
                                  current_read.start_flow, current_read.startSC, current_read.endSC,
                                  current_read.flowIndex,
                                  global_context.flowOrder, variant_identity.underCallPosition, global_context.DEBUG);
  float delta = 0;
  string read_sequence;
  int trunc_delta = 0;
  int evaluate_success = hypEvaluator.calculateHPLikelihood(read_sequence, current_read.start_flow, variant_identity.underCallLength+1,
                         flowPosition, delta, current_read.is_forward_strand);

  if (evaluate_success == 0) {
    trunc_delta = (int)delta*100.0f;

    if (trunc_delta > 0.0f && trunc_delta < MAXSIGDEV)
      flowDist->getHomPolyDist()[trunc_delta]++;
  }
  //add delta to flowDist distribution
  //now evaluate the overall HP length
  flowPosition = 0;
  trunc_delta = 0;
  flowPosition = retrieve_flowpos(current_read.alignment.QueryBases, local_contig_sequence, current_read.is_forward_strand, current_read.ref_aln, current_read.seq_aln, current_read.start_pos,
                                  current_read.start_flow, current_read.startSC, current_read.endSC,
                                  current_read.flowIndex,
                                  global_context.flowOrder, variant_identity.overCallPosition, global_context.DEBUG);
  delta = 0;
  read_sequence = "";
  evaluate_success = hypEvaluator.calculateHPLikelihood(read_sequence, current_read.start_flow, variant_identity.overCallLength-1,
                     flowPosition, delta, current_read.is_forward_strand);
  if (evaluate_success == 0) {
    trunc_delta = (int)delta*100.0f;
    if (trunc_delta > 0 && trunc_delta < MAXSIGDEV)
      flowDist->getCorrPolyDist()[trunc_delta]++;
  }
}

void MultiFlowDist::EvaluateFlowDistForSNP(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
    InputStructures &global_context,
    const string &local_contig_sequence,
    FlowDist *flowDist, int variant_start_pos,
    AlleleIdentity &variant_identity) {

  float refLikelihood = 0.0f;
  float minDelta = 9999999.0f;
  float minDistToRead = 999999.0f;
  float distDelta = 0.0f;
  vector<string> hypotheses;

  int evaluate_splicing = HypothesisSpliceVariant(hypotheses, current_read, variant_identity,
                                   (*(multi_variant.variant))->ref, local_contig_sequence, global_context.DEBUG);

  if (evaluate_splicing==0) {
    int evaluate_success = hypEvaluator.calculateCrudeLikelihood(hypotheses, current_read.start_flow, refLikelihood,  minDelta, minDistToRead,  distDelta);
    if (evaluate_success == 0)
      UpdateFlowDistWithOrdinaryVariant(flowDist,current_read.is_forward_strand, refLikelihood, minDistToRead, distDelta);
  }

  //now if this is a possible FP SNP caused by overall & undercall of two HPs, evaluate the length of each HP length
  if (variant_identity.status.isOverCallUnderCallSNP) {
    ExtraEvalForOverUnderSNP(hypEvaluator, current_read, global_context, local_contig_sequence, flowDist, variant_start_pos, variant_identity);
  }

}

void MultiFlowDist::EvaluateFlowDistForOrdinary(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
    InputStructures &global_context,
    const string &local_contig_sequence,
    FlowDist *flowDist, int variant_start_pos,
    AlleleIdentity &variant_identity) {
  float refLikelihood = 0.0f;
  float minDelta = 9999999.0f;
  float minDistToRead = 999999.0f;
  float distDelta = 0.0f;

  vector<string> hypotheses;
  int evaluate_splicing = HypothesisSpliceVariant(hypotheses, current_read, variant_identity,
                                   (*(multi_variant.variant))->ref, local_contig_sequence, global_context.DEBUG);

  if (evaluate_splicing==0) {
    int evaluate_success = hypEvaluator.calculateCrudeLikelihood(hypotheses, current_read.start_flow,
                           refLikelihood,  minDelta, minDistToRead,  distDelta);

    if (evaluate_success == 0)
      UpdateFlowDistWithOrdinaryVariant(flowDist,current_read.is_forward_strand, refLikelihood, minDistToRead, distDelta);
  }
}

void MultiFlowDist::EvaluateFlowDistForLong(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
    InputStructures &global_context,
    const string &local_contig_sequence,
    FlowDist *flowDist, int variant_start_pos,
    AlleleIdentity &variant_identity) {
  uint32_t flowPosition = 0;

  flowPosition = retrieve_flowpos(current_read.alignment.QueryBases, local_contig_sequence, current_read.is_forward_strand, current_read.ref_aln, current_read.seq_aln, current_read.start_pos,
                                  current_read.start_flow, current_read.startSC, current_read.endSC,
                                  current_read.flowIndex,
                                  global_context.flowOrder, variant_start_pos, global_context.DEBUG);
  if (global_context.DEBUG)
    cout << current_read.is_forward_strand << " " << flowPosition << endl;


  float delta = 0;
  string read_sequence;

  int evaluate_success = hypEvaluator.calculateHPLikelihood(read_sequence, current_read.start_flow, multi_variant.seq_context.my_hp_length.at(0),
                         flowPosition, delta, current_read.is_forward_strand);

  if (global_context.DEBUG) {
    cout << "CalculateHPLikelihood returned " << evaluate_success << endl;
    if (evaluate_success == -1) {
      cerr << "HPEval failed flowPosition = " << flowPosition << " StartFlow = " << current_read.start_flow << endl;
      cerr << "Variant Pos = " << variant_start_pos << endl;
      cerr << "Start Pos = " << current_read.start_pos << endl;
      cerr << "Ref Len = " << multi_variant.seq_context.my_hp_length.at(0) << endl;
      cerr << "RefA : " << current_read.ref_aln << endl;

      cerr << "SeqA : " << current_read.seq_aln << endl;
      cerr << "strand = " << current_read.is_forward_strand << endl;
    }


  }

  if (evaluate_success==0)
    UpdateFlowDistWithLongVariant(flowDist, current_read.is_forward_strand, delta, variant_identity, multi_variant.seq_context);
}

void MultiFlowDist::EvaluateOneFlowDist(HypothesisEvaluator &hypEvaluator, ExtendedReadInfo &current_read,
                                        InputStructures &global_context,
                                        const string &local_contig_sequence,
                                        FlowDist *flowDist, int variant_start_pos,
                                        AlleleIdentity &variant_identity) {
  if (variant_identity.status.isSNP || variant_identity.status.isMNV) {
    EvaluateFlowDistForSNP(hypEvaluator, current_read, global_context, local_contig_sequence, flowDist, variant_start_pos, variant_identity);
  } else
    if (variant_identity.Ordinary()) {
      EvaluateFlowDistForOrdinary(hypEvaluator, current_read, global_context, local_contig_sequence, flowDist, variant_start_pos, variant_identity);
    } else {
      EvaluateFlowDistForLong(hypEvaluator, current_read, global_context, local_contig_sequence, flowDist, variant_start_pos, variant_identity);
    }
}

bool MultiFlowDist::ScoreAllAlleles(vcf::Variant ** candidate_variant, ExtendParameters *parameters,  int DEBUG) {
  bool isFiltered = false;
  bool isNoCall = false;
  //bool isBestAlleleSNP = false;
 // bool isIndelAllelePresent = false;

  FlowDist * flowDist;
  AlleleIdentity variant_identity;
  max_score_allele_index = -1;
  //float maxScore = -1;
  filteredAllelesIndex.resize(0);
  for (unsigned int counter = 0; counter < flowDistVector.size(); counter++) {
    flowDist = flowDistVector[counter];
    variant_identity = multi_variant.allele_identity_vector[counter];
    //if any of the allele is a NoCall (hard to imagine case where only one of alt alleles will be classified as No call) then mark the variant as no call
    if (variant_identity.status.isNoCallVariant)
      isNoCall = true;

    if (variant_identity.status.isSNP || variant_identity.status.isMNV) {
      // ordinary score
      CalculateOrdinaryScore(flowDist, variant_identity,
                             candidate_variant, parameters->my_controls, &isFiltered, DEBUG);
    } else {
      //isIndelAllelePresent = true;
      if (variant_identity.Ordinary()) {
        CalculateOrdinaryScore(flowDist, variant_identity, candidate_variant,
                               parameters->my_controls, &isFiltered, DEBUG);
      } else {
        CalculatePeakFindingScore(flowDist, multi_variant, counter, parameters->my_controls, &isFiltered, DEBUG);
      }
    }

    if (variant_identity.status.isReferenceCall) {
      multi_variant.allele_identity_vector[counter].status.isReferenceCall = true;
    }

 
  }
 
  return(isNoCall);
}

// score the whole flowdist structure
void MultiFlowDist::ScoreFlowDistForVariant(vcf::Variant ** candidate_variant,
    ExtendParameters *parameters,  int DEBUG) {

  ScoreAllAlleles(candidate_variant, parameters, DEBUG);

}

void MultiFlowDist::OutputAlleleToVariant(vcf::Variant ** candidate_variant, ExtendParameters *parameters) {
  DecisionTreeData my_decision;
  vector<VariantBook> summary_stats_vector;
  vector<VariantOutputInfo> summary_info_vector;
  

  if (flowDistVector.size() != multi_variant.allele_identity_vector.size() ) {
    cerr << "FATAL ERROR: Size Of allele vector = " << flowDistVector.size() << " not equal to variant identity vector " << multi_variant.allele_identity_vector.size() << endl;
    cout << (**candidate_variant) << endl;
    exit(-1);
  }

  for (size_t i = 0; i < flowDistVector.size(); i++) {
    summary_stats_vector.push_back(flowDistVector.at(i)->summary_stats);
    summary_info_vector.push_back(flowDistVector.at(i)->summary_info);
    //variant_identity_vector.push_back(multi_variant.variant_identity_vector.at(i));
  }

  my_decision.summary_stats_vector = summary_stats_vector;
  my_decision.summary_info_vector = summary_info_vector;
  //my_decision.variant_identity_vector = variant_identity_vector;
  my_decision.multi_allele = multi_variant;
  my_decision.SetLocalGenotypeCallFromStats(parameters->my_controls.filter_snps.min_allele_freq); // use the "summary stats" to define genotype - this may be done elsewhere differently!!!
  //cout << "Calling filter alleles "<<endl;
  my_decision.DecisionTreeOutputToVariant(candidate_variant, parameters);

}



