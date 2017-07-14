/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include <cassert>
#include <cmath>
#include <iostream>
#include <iostream>
#include <iomanip>
#include <limits>
#include "PhaseSolve.h"
#include "IonErr.h"

#define PHASE_DEBUG 0

PhaseSolve::PhaseSolve() {
  signal.resize(0);
  predictedSignal.resize(0);
  residualSignal.resize(0);
  secondResidualSignal.resize(0);
  correctedSignal.resize(0);
  hpFlow.resize(0);

  residualScale            = true;
  residualScaleMinFlow     = 12;
  residualScaleMaxFlow     = 50;
  residualScaleMinSignal   = MIN_NORMALIZED_SIGNAL_VALUE;
  residualScaleHpWeight.resize(MAX_HOMOPOLYMER_LENGTH,0);
  residualScaleHpWeight[1] = 1;
  residualScaleHpWeight[2] = 0.8;
  residualScaleHpWeight[3] = 0.64;
  residualScaleHpWeight[4] = 0.512;
  multiplier.assign(1,1);
}


unsigned int PhaseSolve::GreedyBaseCall (
  weight_vec_t &          signal,
  unsigned int            nIterations,
  bool                    debug
) {
  return(GreedyBaseCall(
    signal,
    nIterations,
    extendAdvancerFirst,
    droopAdvancerFirst,
    extendAdvancer,
    droopAdvancer,
    debug
  ));
}

unsigned int PhaseSolve::GreedyBaseCall (
  weight_vec_t & _signal,
  unsigned int   nIterations,
  advancer_t   & thisExtendAdvancerFirst,
  advancer_t   & thisDroopAdvancerFirst,
  advancer_t   & thisExtendAdvancer,
  advancer_t   & thisDroopAdvancer,
  bool           debug
) {
  signal = _signal;
  unsigned int nFlow = signal.size();
  unsigned int nFlowPerCycle = flowCycle.size();

  // We require phase advancers to already have been set, so complain if not.
  if(thisExtendAdvancerFirst.size() == 0)
    ION_ABORT("First extend advancer has zero size - forget to set it?");
  if(thisDroopAdvancerFirst.size() == 0)
    ION_ABORT("First droop advancer has zero size - forget to set it?");
  if(thisExtendAdvancer.size() == 0)
    ION_ABORT("Extend advancer has zero size - forget to set it?");
  if(thisDroopAdvancer.size() == 0)
    ION_ABORT("Droop advancer has zero size - forget to set it?");

  // Set sequence to an empty string.
  setSeq(""); // TODO: allow for non-empty initial guess, might allow faster convergence
  setAdvancerContexts(maxAdvances);

  // Initialize vectors used to store history of template configurations
  vector <readLen_vec_t> vPrevPosWeight;
  vector <weight_vec_t> vPrevHpWeight;
  weight_vec_t vPrevDroopedWeight,vPrevIgnoredWeight;
  vector <hpLen_vec_t> vPrevHpLen;
  vector <nuc_vec_t> vPrevHpNuc;
  unsigned int nRewindFlow = signal.size();
  vPrevPosWeight.reserve(nRewindFlow+1);
  vPrevHpWeight.reserve(nRewindFlow+1);
  vPrevDroopedWeight.reserve(nRewindFlow+1);
  vPrevIgnoredWeight.reserve(nRewindFlow+1);
  vPrevHpLen.reserve(nRewindFlow+1);
  vPrevHpNuc.reserve(nRewindFlow+1);

  // These vectors track the edits that we make relative to the guess at the start of each iteration
  vector <bool> edit(nFlow);                    // Indicates if an edit was made
  vector <unsigned int> editHpIndex(nFlow);     // The iHP for which we made the edit
  vector <hpLen_t> editHpLen(nFlow);            // The hpLength to which we edited
  vector <unsigned int> editNextHpIndex(nFlow); // The iHP for the next edit that will be made at or after a given flow

  // These vectors are scratch space for the function PhaseSim::applyFlow
  weight_vec_t  newHpWeight;
  readLen_vec_t newPosWeight;

  hpFlow.assign(nFlow,0);
  predictedSignal.assign(nFlow,0);
  residualSignal.assign(nFlow,0);
  secondResidualSignal.assign(nFlow,0);
  correctedSignal.assign(nFlow,0);
  bool firstIteration=true;
  unsigned int iIteration=0;
  while(iIteration < nIterations) {
    iIteration++;

    // After initial iteration, perform residual scaling if requested.
    if(residualScale && (iIteration > 0)) {
      weight_t newMult = rescale();
      multiplier.push_back(multiplier.back() * newMult);
      if(debug)
        cout << "Rescaling by " << newMult << "\ttotalMult = " << multiplier.back() << endl;
    }

    // Reset templates to beginning of their current sequence and save intial state in the vPrev* vectors
    resetTemplate();
    readLen_vec_t prevPosWeight;
    weight_vec_t  prevHpWeight;
    weight_t      prevDroopedWeight,prevIgnoredWeight;
    hpLen_vec_t   prevHpLen;
    nuc_vec_t     prevHpNuc;
    saveTemplateState(prevHpWeight, prevDroopedWeight, prevIgnoredWeight, prevPosWeight, prevHpLen, prevHpNuc);
    vPrevPosWeight.resize(0);
    vPrevHpWeight.resize(0);
    vPrevDroopedWeight.resize(0);
    vPrevIgnoredWeight.resize(0);
    vPrevHpLen.resize(0);
    vPrevHpNuc.resize(0);
    vPrevPosWeight.push_back(prevPosWeight);
    vPrevHpWeight.push_back(prevHpWeight);
    vPrevDroopedWeight.push_back(prevDroopedWeight);
    vPrevIgnoredWeight.push_back(prevIgnoredWeight);
    vPrevHpLen.push_back(prevHpLen);
    vPrevHpNuc.push_back(prevHpNuc);

    // Reset edit tracking as we start this new iteration
    edit.assign(nFlow,false);
    editHpIndex.assign(nFlow,0);
    editHpLen.assign(nFlow,'N');
    editNextHpIndex.assign(nFlow,0);

    // Initialize vars that we use to track if we reached the apparent template end
    vector <bool> finishedNuc(N_NUCLEOTIDES,false);

    // Iterate over all flows
    bool testOnly = true;
    bool forReal = false;
    bool firstCycle = true;
    for(unsigned int iFlow=0, iHP=0; iFlow < nFlow; iFlow++) {
      if(iFlow >= nFlowPerCycle)
        firstCycle = false;

      nuc_t testNuc = flowCycle[iFlow % nFlowPerCycle];

      // Iterate over the possible HP lengths to find the best for the HP at index iHP
      weight_t testSignal = 0;
      weight_t smallestDelta = FLT_MAX;
      weight_t smallestAbsDelta = FLT_MAX;
      weight_t secondSmallestDelta = FLT_MAX;
      weight_t secondSmallestAbsDelta = FLT_MAX;
      hpLen_t bestHpLen = 0;
      //hpLen_t secondBestHpLen = 0;
      // tempHpLen records the currently-assumed hpLength at flow iFlow.
      // hpFlow records the state of the sequence guess going into the iteration over all flows.
      // hpFlow won't be updated until after the iteration over all flows.
      hpLen_t tempHpLen = hpFlow[iFlow];
      //weight_t predictedSigZeroMer = -1;
      weight_t predictedSigOneMer  = -1;
      weight_t predictedSigTwoMer  = -1;
      bool alreadySeenNegativeDelta = false; // used to control early exit from loop over HPs
      for(hpLen_t testHpLen=0; testHpLen < MAX_HOMOPOLYMER_LENGTH; testHpLen++) {
        bool hpInsertion = (testHpLen > 0  && tempHpLen == 0);
        bool hpDeletion  = (testHpLen == 0 && tempHpLen >  0);

        // Some 0mers internal to the sequence are disallowed because they would imply changes in
        // earlier flows that were already decided-upon.  If a 0mer is disallowed, just skip to the
        // next considered HP length.
        if((testHpLen==0) && ((iHP+1) < hpNuc.size())) {
          bool illegalZeroMer = false;
          nuc_t nextTemplateNuc = hpDeletion ? hpNuc[iHP+1] : hpNuc[iHP];
          if((iHP > 0) && (hpNuc[iHP-1] == nextTemplateNuc)) {
            // 0mer would change the length of the previously determined incorporating flow
            illegalZeroMer = true;
          } else if(iFlow > 0) {
            // Look back over any immediately-previous runs of 0mer flows to make sure we're not
            // sliding back a positive nuc into any of them
            for(unsigned int flowsBack=1; flowsBack <= iFlow; flowsBack++) {
              unsigned int prevFlow = iFlow-flowsBack;
              hpLen_t prevHp = (edit[prevFlow]) ? editHpLen[prevFlow] : hpFlow[prevFlow];
              if(prevHp > 0) {
                break;
              } else {
                nuc_t prevFlowedNuc = flowCycle[prevFlow % nFlowPerCycle];
                if(nextTemplateNuc == prevFlowedNuc) {
                  illegalZeroMer = true;
                  break;
                }
              }
            }
          }
          if(illegalZeroMer) {
            if(debug)
              cout << "  Skipping illegal 0mer, iHP = " << iHP << ", flow = " << iFlow << endl;
            continue;
          }
        }

        // After the first iteration indels in the runs of HPs require that we rewind and re-apply some flows
        if( (!firstIteration) && (iFlow > 0) && (hpInsertion || hpDeletion) ) {
          redoFlow(
            iHP, iFlow, iFlow, testHpLen,
            vPrevPosWeight, vPrevHpWeight, vPrevDroopedWeight, vPrevIgnoredWeight, vPrevHpLen, vPrevHpNuc,
            thisExtendAdvancerFirst, thisDroopAdvancerFirst, thisExtendAdvancer, thisDroopAdvancer,
            edit, editHpIndex, editHpLen, editNextHpIndex, maxAdvances, false, debug, newHpWeight, newPosWeight
          );
        } else {
#if PHASE_DEBUG
          assert(hpWeightOK());
#endif
          updateSeq(iHP,testHpLen,testNuc,tempHpLen,maxAdvances);
        }
        tempHpLen = testHpLen;
#if PHASE_DEBUG
        assert(hpWeightOK());
#endif

        // Apply the flow, test mode
#if PHASE_DEBUG
        assert(hpWeightOK());
#endif
        testSignal = applyFlow(iFlow, (firstCycle ? thisExtendAdvancerFirst : thisExtendAdvancer), (firstCycle ? thisDroopAdvancerFirst : thisDroopAdvancer), testOnly, newHpWeight, newPosWeight);
        //if(testHpLen==0) {
        //  predictedSigZeroMer = testSignal;
        //} else
        if(testHpLen==1) {
          predictedSigOneMer = testSignal;
        } else if(testHpLen==2) {
          predictedSigTwoMer = testSignal;
        }
        if(debug) {
          string testSeq;
          getSeq(testSeq);
          cout << "flow " << iFlow << "\tit " << iIteration << "\tiHP " << iHP << "\tmin,max HP = " << posWeight.front() << "," << posWeight.back() << "\ttestHpLen " << (int) testHpLen << "\ttestSig " << testSignal << "\tstring " << testSeq << "\n";
        }
#if PHASE_DEBUG
        assert(hpWeightOK());
#endif

        // Check if this is the best fit yet seen, if so then store it.
        weight_t delta = signal[iFlow] - testSignal;
        weight_t absDelta = abs(delta);
        if(absDelta < smallestAbsDelta) {
          // move the best into the second-best slot
          secondSmallestDelta = smallestDelta;
          secondSmallestAbsDelta = smallestAbsDelta;
          //secondBestHpLen = bestHpLen;
          // store the new best
          smallestDelta = delta;
          smallestAbsDelta = absDelta;
          bestHpLen = testHpLen;
        } else if(absDelta < secondSmallestAbsDelta) {
          // update the second-best stats
          secondSmallestDelta = delta;
          secondSmallestAbsDelta = absDelta;
          //secondBestHpLen = testHpLen;
        }

        // If the nuc to test is one that we have flowed already and we haven't seen any incorporations
        // since then break out after evaluating a 0mer because we don't want to update the previous guess.
        if(finishedNuc[testNuc])
          break;

        // If we're over-predicting then break out of the loop, no need to continue.
        // Need to have seen at least the candidate 0mer, 1mer and 2mer before making this decision
        // because there are some situations where putting in a 0mer can bring forward another big
        // HP of the same nuc, leading to the situation where the 0mer actually predicts a big signal,
        if(testHpLen > 0 && delta < 0) {
          if(alreadySeenNegativeDelta) {
            break;
          } else {
            alreadySeenNegativeDelta = true;
          }
        }
      }

      // Rewind and redo recent flows if necessary
      bool hpInsertion = (bestHpLen > 0  && tempHpLen == 0);
      bool hpDeletion  = (bestHpLen == 0 && tempHpLen >  0);
      if( (!firstIteration) && (iFlow > 0) && (hpInsertion || hpDeletion) ) {
        redoFlow(
          iHP, iFlow, iFlow, bestHpLen,
          vPrevPosWeight, vPrevHpWeight, vPrevDroopedWeight, vPrevIgnoredWeight, vPrevHpLen, vPrevHpNuc,
          thisExtendAdvancerFirst, thisDroopAdvancerFirst, thisExtendAdvancer, thisDroopAdvancer,
          edit, editHpIndex, editHpLen, editNextHpIndex, maxAdvances, true, debug, newHpWeight, newPosWeight
        );
      } else {
        updateSeq(iHP,bestHpLen,testNuc,tempHpLen,maxAdvances);
      }

      if(debug) {
        string testSeq;
        getSeq(testSeq);
        cout << "  about to store flow " << iFlow << "\tmin,max HP = " << posWeight.front() << "," << posWeight.back() << "\tseq = " << testSeq << "\n";
      }

      // Apply the current flow
#if PHASE_DEBUG
      assert(hpWeightOK());
#endif
      predictedSignal[iFlow] = applyFlow(iFlow, (firstCycle ? thisExtendAdvancerFirst : thisExtendAdvancer), (firstCycle ? thisDroopAdvancerFirst : thisDroopAdvancer), forReal, newHpWeight, newPosWeight);
#if PHASE_DEBUG
      assert(hpWeightOK());
#endif
      residualSignal[iFlow] = smallestDelta;
      secondResidualSignal[iFlow] = secondSmallestDelta;
      weight_t predictedDelta = predictedSigTwoMer - predictedSigOneMer;
      if(predictedDelta <= numeric_limits<weight_t>::epsilon()) {
        predictedDelta = 1.0; // avoid divide-by-zero
      }
      weight_t adjustment = residualSignal[iFlow]/predictedDelta;
      if(abs(adjustment) >= 0.5) {
        adjustment = (adjustment > 0) ? 0.49 : -0.49;  // ensure corrected value always rounds to the guessed incorporation
      }
      correctedSignal[iFlow] = ((weight_t) bestHpLen) + adjustment;

      // Store the chosen hp Length
#if PHASE_DEBUG
      assert(hpWeightOK());
      if(debug) {
        string testSeq;
        getSeq(testSeq);
        cout << "STORE: flow " << iFlow << "\thpLen " << (int) bestHpLen  << "\tmin,max HP = " << posWeight.front() << "," << posWeight.back() << "\tdelta " << setiosflags(ios::fixed) << setprecision(6) << smallestAbsDelta << "\tseq " << testSeq << "\n";
      }
#endif

      // Update edit vectors if we made an edit
      if(bestHpLen != hpFlow[iFlow]) {
        edit[iFlow] = true;
        editHpIndex[iFlow] = iHP;
        editHpLen[iFlow] = bestHpLen;
        editNextHpIndex[iFlow] = iHP;
        for(int iPrevFlow=iFlow-1; ; iPrevFlow--) {
          if( (iPrevFlow < 0) || (editNextHpIndex[iPrevFlow]!=0) )
            break;
          else
            editNextHpIndex[iPrevFlow] = iHP;
        }
      }
      if(bestHpLen > 0)
        iHP++;

      if(!firstIteration) {
        // Save the state
        saveTemplateState(prevHpWeight, prevDroopedWeight, prevIgnoredWeight, prevPosWeight, prevHpLen, prevHpNuc);
        vPrevPosWeight.push_back(posWeight);
        vPrevHpWeight.push_back(prevHpWeight);
        vPrevDroopedWeight.push_back(prevDroopedWeight);
        vPrevIgnoredWeight.push_back(prevIgnoredWeight);
        vPrevHpLen.push_back(prevHpLen);
        vPrevHpNuc.push_back(prevHpNuc);
      }

#if PHASE_DEBUG
      // Check to make sure the nuc sequence matches what we have in flows
      hpLen_t thisFlowLen = (edit[iFlow]) ? editHpLen[iFlow] : hpFlow[iFlow];
      hpLen_vec_t tempFlow;
      computeSeqFlow(hpLen, hpNuc, flowCycle, tempFlow);
      string testSeq;
      getSeq(testSeq);
      if(tempFlow.size() > iFlow) {
        assert(tempFlow[iFlow]==thisFlowLen);
      } else {
        assert(0==thisFlowLen);
      }
#endif

      // Bail out if we've apparently reached the end
      if(bestHpLen > 0) {
        for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; ++iNuc)
          finishedNuc[iNuc] = (iNuc==testNuc);
      } else {
        // Mark the nuc we just tested as finished
        finishedNuc[testNuc] = true;

        // Check if we have now finished all 4 nucs
        bool finished = true;
        for(unsigned int iNuc=0; iNuc < N_NUCLEOTIDES; ++iNuc) {
          if(!finishedNuc[iNuc]) {
            finished = false;
            break;
          }
        }
     
        // If we finished all 4 nucs then break out of this iteration over flows
        if(finished)
          break;
      }
    }

    // Count how many flows were edited
    unsigned int nEditedFlows = 0;
    for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
      if(edit[iFlow])
        nEditedFlows++;
    }

    // update the current flow-sequence guess
    for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
      if(edit[iFlow])
        hpFlow[iFlow] = editHpLen[iFlow];
    }

#if PHASE_DEBUG
    // Pedantic checking to make sure we've got a legal flow sequence
    hpLen_vec_t      tempFlow;
    computeSeqFlow(hpLen, hpNuc, flowCycle, tempFlow);
    if(tempFlow.size() < nFlow) {
      for(unsigned int zeroFlow=tempFlow.size(); zeroFlow < nFlow; zeroFlow++)
        tempFlow.push_back(0);
    }
    for(unsigned int iFlow=0; iFlow < nFlow; iFlow++) {
      assert(hpFlow[iFlow]==tempFlow[iFlow]);
    }
#endif

    firstIteration = false;

    if(nEditedFlows==0)
      break;
  }

  return(iIteration);
}

weight_t PhaseSolve::rescale(void) {
  unsigned int maxFlow = std::min(residualScaleMaxFlow, (unsigned int) signal.size()-1);
  unsigned int minFlow = std::min(residualScaleMinFlow, maxFlow);
  hpLen_t nWeight = (hpLen_t) residualScaleHpWeight.size();
  weight_t residual_numerator=0;
  weight_t residual_denominator=0;
  for(unsigned int iFlow=minFlow; iFlow<=maxFlow; iFlow++) {
  	if(hpFlow[iFlow] < nWeight) {
  		if(isnan(signal[iFlow]) || isnan(predictedSignal[iFlow]))
  			continue;
  		weight_t meas = std::max(signal[iFlow],residualScaleMinSignal);
  		weight_t pred = std::max(predictedSignal[iFlow],residualScaleMinSignal);
  		weight_t logRatio = log2(meas/pred);
  		weight_t thisWeight = residualScaleHpWeight[hpFlow[iFlow]];
  		residual_numerator += logRatio * thisWeight;
  		residual_denominator += thisWeight;
  	}
  }
  if(residual_denominator > 0) {
  	weight_t thisMult = pow(2.0,-1*residual_numerator/residual_denominator);
  	for(unsigned int iFlow=0; iFlow<signal.size(); iFlow++)
  		signal[iFlow] *= thisMult;
  	return(thisMult);
  } else {
  	return(1.0);
  }
}

void PhaseSolve::updateSeq(
  unsigned int iHP,
  hpLen_t newHpLen,
  nuc_t newNuc,
  hpLen_t oldHpLen,
  hpLen_t maxAdvances
) {
#if PHASE_DEBUG
  assert(hpWeightOK());
#endif
  if(newHpLen != oldHpLen) {
    bool insertion = (newHpLen >  0 && oldHpLen == 0);
    bool deletion  = (newHpLen == 0 && oldHpLen >  0);

    // Update hpNuc & hpLen
    if(insertion) {
      hpLen.insert(hpLen.begin()+iHP,newHpLen);
      hpNuc.insert(hpNuc.begin()+iHP,newNuc);
      hpWeight.insert(hpWeight.begin()+iHP+1,0);
#if PHASE_DEBUG
      assert(hpWeightOK());
#endif
    } else if(deletion) {
      hpLen.erase(hpLen.begin()+iHP);
      hpNuc.erase(hpNuc.begin()+iHP);
#if PHASE_DEBUG
      assert(hpWeight[iHP+1]      <= numeric_limits<weight_t>::epsilon());
#endif
      hpWeight.erase(hpWeight.begin()+iHP+1);
#if PHASE_DEBUG
      assert(hpWeightOK());
#endif
    } else {
      // Change in HP length - only need to change hpLen
      hpLen[iHP] = newHpLen;
    }

    // Update advancers.  For now we re-do everything on any indel
    // but for speed we should be able to update just a small section
    // in the region of the indel
    //if(insertion || deletion)
    //  setAdvancerContexts(maxAdvances);
  }

#if PHASE_DEBUG
  unsigned int nHP = hpNuc.size();
  unsigned int nWeight = hpWeight.size();
  assert(nHP+1 == nWeight);
#endif

  setAdvancerContexts(maxAdvances);
}

void PhaseSolve::redoFlow(
  unsigned int             iHP,
  unsigned int             iFlow,
  unsigned int             editFlow,
  hpLen_t                  testHpLen,
  vector <readLen_vec_t> & vPrevPosWeight,
  vector <weight_vec_t>  & vPrevHpWeight,
  weight_vec_t           & vPrevDroopedWeight,
  weight_vec_t           & vPrevIgnoredWeight,
  vector <hpLen_vec_t>   & vPrevHpLen,
  vector <nuc_vec_t>     & vPrevHpNuc,
  advancer_t             & thisExtendAdvancerFirst,
  advancer_t             & thisDroopAdvancerFirst,
  advancer_t             & thisExtendAdvancer,
  advancer_t             & thisDroopAdvancer,
  vector <bool>          & edit,
  vector <unsigned int>  & editHpIndex,
  vector <hpLen_t>       & editHpLen,
  vector <unsigned int>  & editNextHpIndex,
  hpLen_t                  maxAdvances,
  bool                     storeSignal,
  bool                     debug,
  weight_vec_t           & newHpWeight,
  readLen_vec_t          & newPosWeight
) {
  unsigned int nFlowPerCycle = flowCycle.size();
  nuc_t testNuc = flowCycle[editFlow % nFlowPerCycle];

  vector <readLen_vec_t>::iterator vPrevPosWeightIt     = vPrevPosWeight.end();
  vector <weight_vec_t>::iterator  vPrevHpWeightIt      = vPrevHpWeight.end();
  weight_vec_t::iterator           vPrevDroopedWeightIt = vPrevDroopedWeight.end();
  weight_vec_t::iterator           vPrevIgnoredWeightIt = vPrevIgnoredWeight.end();
  vector <hpLen_vec_t>::iterator   vPrevHpLenIt         = vPrevHpLen.end();
  vector <nuc_vec_t>::iterator     vPrevHpNucIt         = vPrevHpNuc.end();
  if(debug) {
    cout << "previous posWeight vals [" << vPrevPosWeight.size() << "]" << endl;
    while(vPrevPosWeightIt != vPrevPosWeight.begin()) {
      vPrevPosWeightIt--;
      for(unsigned int i=0; i < (*vPrevPosWeightIt).size(); i++)
        cout << " " << (*vPrevPosWeightIt)[i];
      cout << endl;
    }
    vPrevPosWeightIt = vPrevPosWeight.end();
  }
  unsigned int nFlowsToRewind=0;
  unsigned int tempFlow=iFlow;
  while( vPrevPosWeightIt != vPrevPosWeight.begin() ) {
    vPrevPosWeightIt--;
    vPrevHpWeightIt--;
    vPrevDroopedWeightIt--;
    vPrevIgnoredWeightIt--;
    vPrevHpLenIt--;
    vPrevHpNucIt--;
    // Keep rewinding flows until the start, or until the following two properties both hold:
    //   1 - The longest template is maxAdvances shorter than the position that we want to edit
    //   2 - The longest template is maxAdvances shorter than any other positions for which edits
    //       will need to be reapplied at or after the flow we wind back to
    // The second criterion is where we need the editNextHpIndex variable.
    // TODO: Make this function "rewrite history" to make history consistent
    // with the current greedy guess.  Would allow for skipping criterion 2 above
    // and should make things faster.
    unsigned int limit1 = std::max( (int) iHP - (int) maxAdvances - 1, 0 );
    unsigned int limit2 = std::max( (int) editNextHpIndex[tempFlow] - (int) maxAdvances - 1, 0 );
    if( ((*vPrevPosWeightIt).back() <= limit1) && ((editNextHpIndex[tempFlow] == 0) || ((*vPrevPosWeightIt).back() <= limit2)) )
      break;
    nFlowsToRewind++;
    tempFlow--;
  }

  if(debug) {
    unsigned int maxRewoundHpLen = std::max(((int) iHP) - 1 - (int) maxAdvances, 0);
    cout << " rewinding " << nFlowsToRewind << " to start of flow " << iFlow-nFlowsToRewind << "  cur maxHpLen is " << posWeight.back() << " target max is " << maxRewoundHpLen << "\n";
  }
  // Rewind back nFlowsToRewind flows
  //nFlowsToRewind = iFlow;
  //resetTemplate(vPrevHpWeight.front(), vPrevDroopedWeight.front(), vPrevPosWeight.front(), vPrevHpLen.front(), vPrevHpNuc.front());
  resetTemplate(*vPrevHpWeightIt, *vPrevDroopedWeightIt, *vPrevIgnoredWeightIt, *vPrevPosWeightIt, *vPrevHpLenIt, *vPrevHpNucIt);
  if(debug) {
    cout << "   after rewinding  maxHpLen is " << posWeight.back() << " or " << (*vPrevPosWeightIt).back() << "\n";
  }
  if(debug) {
    string testSeq;
    getSeq(testSeq);
    cout << "  seq1 " << testSeq << "\n";
  }
#if PHASE_DEBUG
  assert(hpWeightOK());
#endif

  // Re-apply sequence edits made since the rewind point
  for(unsigned int iRedoFlow=iFlow-nFlowsToRewind; iRedoFlow < iFlow; iRedoFlow++) {
    if(edit[iRedoFlow]) {
      nuc_t redoNuc = flowCycle[iRedoFlow % nFlowPerCycle];
      updateSeq(editHpIndex[iRedoFlow],editHpLen[iRedoFlow],redoNuc,hpFlow[iRedoFlow],maxAdvances);
    }
  }
  if(debug) {
    string testSeq;
    getSeq(testSeq);
    cout << "  seq2 " << testSeq << "\n";
  }

  // Apply the edit being tested
  updateSeq(iHP,testHpLen,testNuc,hpFlow[editFlow],maxAdvances);
#if PHASE_DEBUG
  assert(hpWeightOK());
#endif
  if(debug) {
    cout << "  editFlow = " << editFlow << "  edited = " << (int)edit[editFlow] << "  editHpLen = " << (int)editHpLen[editFlow] << "  hpFlow = " << (int)hpFlow[editFlow] << endl;
    string testSeq;
    getSeq(testSeq);
    cout << "  seq3 " << testSeq << "\tlen " << testSeq.length() << "\tnHP " << hpLen.size() << "\n";
  }
  // Re-apply the flows up to but not including the edited flow
  bool testOnly = false;
  for(unsigned int iRedoFlow=editFlow-nFlowsToRewind; iRedoFlow < editFlow; iRedoFlow++) {
#if PHASE_DEBUG
    assert(posWeight.back() <= hpLen.size());
#endif
    bool firstCycleOfRedo = (iRedoFlow < nFlowPerCycle);
    weight_t thisSig = applyFlow(iRedoFlow, (firstCycleOfRedo ? thisExtendAdvancerFirst : thisExtendAdvancer), (firstCycleOfRedo ? thisDroopAdvancerFirst : thisDroopAdvancer), testOnly, newHpWeight, newPosWeight);
    if(storeSignal) {
      predictedSignal[iRedoFlow] = thisSig;
      residualSignal[iRedoFlow] = signal[iRedoFlow] - thisSig;
    }
    if(debug) {
      cout << "  reapplying flow " << iRedoFlow << " min,max HP = " << posWeight.front() << "," << posWeight.back() << endl;
    }
  }
#if PHASE_DEBUG
  assert(hpWeightOK());
#endif
}

void PhaseSolve::GetBaseFlowIndex(vector<uint16_t> &baseFlowIndex) {
  baseFlowIndex.clear();
  unsigned int prev_used_flow = 0;
  for (unsigned int iFlow=0; iFlow < hpFlow.size(); iFlow++) {
    hpLen_t hpLen = hpFlow[iFlow];
    while (hpLen > 0) {
      baseFlowIndex.push_back(1 + iFlow - prev_used_flow);
      prev_used_flow = iFlow + 1;
      hpLen--;
    }
  }
}

void PhaseSolve::GetCorrectedSignal(weight_vec_t & _correctedSignal) {
  _correctedSignal = correctedSignal;
}
