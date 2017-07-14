/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// State-based predictive CAFIE solver
// (c) 2008 Ion Torrent Systems, Inc.
// Author: Mel Davey
//

#include <assert.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <sstream>

#include "CafieSolver.h"
#include "DroopFit.h"
#include "CFIEDRFit.h"

#include "Stats.h"
#include "Utils.h"

// #define ACCURATE

int CafieSolver::numIonogramGrids = 0;
IonogramGrid CafieSolver::ionogramGrid[100];
int CafieSolver::instances = 0;

CafieSolver::CafieSolver()
{
	strandCount = 10000;
	numTemplateBases = 0;
	testLen = 0;
	meanCaf = 0.04;
	meanIe = 0.04;
	numFlows = 0;
	numCalls = 0;
	seqString = NULL;
	memset (measured, 0, sizeof(double)*MAX_FLOWS);
	currentTestSequence[0] = 0;
	//instances++;  // not thread safe, and variable is unused anyway!
	flowOrder = NULL;
	flowOrderIndex = NULL;
	multiplier = 1;
}

CafieSolver::~CafieSolver()
{
#if 0
	if (seqString)
		delete [] seqString;
	instances--;
	if (instances == 0) {
		int i;
		for(i=0;i<numIonogramGrids;i++) {
			int k;
			for(k=0;k<ionogramGrid[i].cfNum * ionogramGrid[i].ieNum;k++) {
				free(ionogramGrid[i].predictedValue[k]);
			}
		}
		free(ionogramGrid[i].predictedValue);
		numIonogramGrids = 0;
	}
#endif

	if (flowOrder)
		free(flowOrder);
	flowOrder = NULL;
	if (flowOrderIndex)
		delete [] flowOrderIndex;
	if (seqString)
		delete [] seqString;

}


void CafieSolver::AddState(ExtensionSim *extSim, int strands, int state)
{
if (extSim->numStates >= MAX_STATES) {
	printf("ERROR!  Max states exceeded!\n");
	return;
}
#ifndef NO_MALLOC
	extSim->states = (ExtensionState *)realloc(extSim->states, sizeof(ExtensionState) * (extSim->numStates+1));
#endif /* NO_MALLOC */
	extSim->states[extSim->numStates].strands = strands;
	extSim->states[extSim->numStates].state = state;
	extSim->numStates++;
}

void inline CafieSolver::CompactStates(ExtensionSim *extSim)
{
	int numStates = extSim->numStates;
	int sparseStateList[MAX_STATES];
	memset(sparseStateList, 0, sizeof(sparseStateList));
	int state;
	int i;
	int minState = MAX_STATES;
	int maxState = 0;
	for(i=0;i<numStates;i++) {
		state = extSim->states[i].state;
		sparseStateList[state] += extSim->states[i].strands;
		if (state < minState)
			minState = state;
		if (state > maxState)
			maxState = state;
	}

#ifndef NO_MALLOC
	int j = 0;
	for(i=minState;i<=maxState;i++) {
		if (sparseStateList[i] > 0)
			j++;
	}
	extSim->states = (ExtensionState *)realloc(extSim->states, sizeof(ExtensionState) * j);
#endif /* NO_MALLOC */

	extSim->numStates = 0;
	for(i=minState;i<=maxState;i++) {
		if (sparseStateList[i] > 0) {
			extSim->states[extSim->numStates].state = i;
			extSim->states[extSim->numStates].strands = sparseStateList[i];
			extSim->numStates++;
		}
	}
}

void CafieSolver::DistToBaseInit()
{
	int flowNum;
	int i;
	char base;
	int baseID;
	char baseList[4] = {'A', 'C', 'G', 'T'};
	for(baseID=0;baseID<4;baseID++) {
		base = baseList[baseID];
		for(flowNum=0;flowNum<MAX_FLOWS;flowNum++) {
			Dist[baseID][flowNum] = -1;
			for(i=0;(i<numFlowsPerCycle) && (flowNum-i >= 0);i++) {
				if (flowOrder[(flowNum-i)%numFlowsPerCycle] == base) {
					Dist[baseID][flowNum] = i;
					break;
				}
			}
		}
	}
		
}

// DistToBase - returns how many flows we need to search backward in time with to find a match in the flow order to the input base
//              returns -1 when we can't find an earlier flow that matches
inline int CafieSolver::DistToBase(char base, int flowNum)
{
/*
	int i;
	for(i=0;(i<numFlowsPerCycle) && (flowNum-i >= 0);i++) {
		if (flowOrder[(flowNum-i)%numFlowsPerCycle] == base)
			return i;
	}
	return -1;
*/

	int baseID = 0;
	if (base == 'C') baseID = 1;
	else if (base == 'G') baseID = 2;
	else if (base == 'T') baseID = 3;
	return Dist[baseID][flowNum];
}

double CafieSolver::ApplyReagent(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool testOnly)
{
	double hpSignal[MAX_MER];
	double sigMult=1.0;

	for(int i=0; i < MAX_MER; i++)
		hpSignal[i] = i;
	
	return(ApplyReagent(extSim,dnaTemplate,numBases,flowNum,ie,caf,dr,testOnly,hpSignal,MAX_MER,sigMult));
}

double CafieSolver::ApplyReagent(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool testOnly, double *hpSignal, int nHpSignal, double sigMult)
{
	ExtensionSim *localSim = &_localSim;
	if (testOnly)
		CopyExtensionSim(extSim, localSim);
	else
		localSim = extSim;

	double signal = 0.0;

	int i;
	// int nuc;

	double nucStrength[4];
	nucStrength[0] = 1.0;
	for(i=1;i<4;i++)
		nucStrength[i] = nucStrength[i-1] * caf;

	// nuc = (r%4);

	int num = localSim->numStates; // store how many states we will examine now, since we will dynamically grow as we traverse!
	for(i=0;i<num;i++) {
		if (localSim->states[i].state < numBases) {
			ExtensionSim nucSim;
			nucSim.numStates = 1;
			nucSim.states[0].state = localSim->states[i].state;
			nucSim.states[0].strands = localSim->states[i].strands;
			int k = 0;
			while (k < nucSim.numStates) {
				if (nucSim.states[k].state < numBases) {
				int distToBase = DistToBase(dnaTemplate[nucSim.states[k].state].base, flowNum);
				if (distToBase >= 0 && distToBase < 4) { // has to be 0 or greater, and if too large, we just ignore as the contribution would be very very small
					int strandsAvail = nucSim.states[k].strands * nucStrength[distToBase];
					int howManyDroop = strandsAvail * dr; // MGD - per JD, right now we expect a 1-mer up to a 5-mer to have the same processivity, so no need to scale by count, and in the future this will get better with pH change.
					int howManyIncomplete = (strandsAvail-howManyDroop) * ie;
					int strandsToAdvance = strandsAvail - howManyIncomplete - howManyDroop;
					if (strandsToAdvance >= 1) { // if at least one strand advances, add a new state for it and add to our signal
						AddState(&nucSim, strandsToAdvance, nucSim.states[k].state+1);
						assert(dnaTemplate[nucSim.states[k].state].count < nHpSignal);
						signal += sigMult * strandsToAdvance * hpSignal[dnaTemplate[nucSim.states[k].state].count];
						nucSim.states[k].strands -= (strandsToAdvance+howManyDroop);
						// MGD note:  as we advance strands, we use up nuc, may want to reduce nucStrength[distToBase] to model this, otherwise could get a runoff of any length with caf == 1.0, but clearly there is not enough nuc concentration to do this
					}
				}
				}
				k++;
			};
			// cleanup the local nuc sim, and add its new states to the primary sim
			CompactStates(&nucSim);
/*
			double totalStrands = 0;
			for(k=0;k<nucSim.numStates;k++) {
				AddState(localSim, nucSim.states[k].strands, nucSim.states[k].state);
				totalStrands += nucSim.states[k].strands;
			}
			localSim->states[i].strands -= totalStrands;
*/
			localSim->states[i].strands = 0;
			for(k=0;k<nucSim.numStates;k++) {
				AddState(localSim, nucSim.states[k].strands, nucSim.states[k].state);
			}
		}
	}

	signal /= strandCount;
	return signal;
}

double CafieSolver::ApplyReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool applyCAF, bool testOnly)
{
	double hpSignal[MAX_MER];
	double sigMult=1.0;

	for(int i=0; i < MAX_MER; i++)
		hpSignal[i] = i;
	
	return(ApplyReagentFast(extSim, dnaTemplate, numBases, flowNum, ie, caf, dr, applyCAF, testOnly, hpSignal, MAX_MER, sigMult));
}

double CafieSolver::ApplyReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool applyCAF, bool testOnly, double *hpSignal, int nHpSignal, double sigMult)
{
	// static double ieMult[9] = {1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
	// static double ieMult[9] = {1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4};
	// static double ieMult[9] = {1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	// static double ieMult[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	// const double ieMult = 0.0;

	ExtensionSim *localSim = &_localSim;
	if (testOnly)
		CopyExtensionSim(extSim, localSim);
	else
		localSim = extSim;

	// search all states, any with matching (modulo 4) states will extend
	double signal = 0.0;
	int i;
	int num = localSim->numStates; // store how many states we will examine now, since we will dynamically grow as we traverse!
	int howManyDroop, howManyIncomplete, howManyAdvance;
	for(i=0;i<num;i++) {
		if (localSim->states[i].state < numBases) {
			if (dnaTemplate[localSim->states[i].state].base == flowOrder[flowNum%numFlowsPerCycle]) { // this one extends
				howManyDroop = localSim->states[i].strands * dr; // MGD - we are not scaling droop with HP now per JD.  * dnaTemplate[localSim->states[i].state].count;
				howManyIncomplete = (localSim->states[i].strands-howManyDroop) * ie;
				howManyAdvance = localSim->states[i].strands - howManyIncomplete - howManyDroop;
				assert(dnaTemplate[localSim->states[i].state].count < nHpSignal);
				signal += howManyAdvance * hpSignal[dnaTemplate[localSim->states[i].state].count] * sigMult;
				localSim->states[i].state++;
				localSim->states[i].strands -= (howManyIncomplete+howManyDroop);
				if (howManyIncomplete > 0)
					AddState(localSim, howManyIncomplete, localSim->states[i].state-1);
			}

			if (applyCAF && (localSim->states[i].state < numBases)) { // Normally, since we have already 'called' bases, we can model carry-forward, but not past where we already have calls, so its then questionable where to put this signal
				int base_carry = flowOrder[(flowNum-1+numFlowsPerCycle)%numFlowsPerCycle]; // the last reagent - note: I'm assuming that this is the only reagent, the one from two ago would be 0.02 * 0.02 in quantity (roughly) so its negligable (1/2500th of our signal)
				if (dnaTemplate[localSim->states[i].state].base == base_carry) { // this one extends due to carry forward
					int howMany = localSim->states[i].strands * caf;
					// howManyDroop = howMany * dr; // MGD - no HP droop scaling per JD  * dnaTemplate[localSim->states[i].state].count;
					howManyDroop = localSim->states[i].strands * caf * dr; // MGD - no HP droop scaling per JD  * dnaTemplate[localSim->states[i].state].count;
					howMany -= howManyDroop;
					if (howMany >= 1) {
						AddState(localSim, howMany, localSim->states[i].state+1);
						localSim->states[i].strands -= (howMany+howManyDroop);
						assert(dnaTemplate[localSim->states[i].state].count < nHpSignal);
						signal += howMany * hpSignal[dnaTemplate[localSim->states[i].state].count] * sigMult;

#ifdef CARRY_FORWARD_2
						if (localSim->states[i].state+1 < numBases) {
							if (dnaTemplate[localSim->states[i].state+1].base == flowOrder[flowNum%numFlowsPerCycle]) { // so now the portion that carried forward can cause the original nuke to extend it too!
								int howMany2 = howMany * ie;
								// if (howMany2 < 1.0) // here, when we are down to 1/10000th of our signal, we just drop it
									// howMany2 = 0.0;
								int howMany3 = howMany - howMany2;
								localSim->states[localSim->numStates-1].strands -= howMany3; // remove these strands from the latest bin where we just advanced them to above
								AddState(localSim, howMany3, localSim->states[i].state+2);
								assert(dnaTemplate[localSim->states[i].state+1].count < nHpSignal);
								signal += howMany3 * hpSignal[dnaTemplate[localSim->states[i].state+1].count] * sigMult;
							}
						}
#endif /* CARRY_FORWARD_2 */
					}
				}
			}
		}
	}


	signal /= strandCount;
	// printf("Measured signal is: %.2lf\n", signal);
	return signal;
}

#ifdef NOT_USED
double CafieSolver::ApplyReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool applyCAF, double *hpSignal, int nHpSignal, double sigMult)
{
	// static double ieMult[9] = {1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
	// static double ieMult[9] = {1.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4};
	// static double ieMult[9] = {1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	// static double ieMult[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	// const double ieMult = 0.0;

	// search all states, any with matching (modulo 4) states will extend
	double signal = 0.0;
	int i;
	int num = extSim->numStates; // store how many states we will examine now, since we will dynamically grow as we traverse!
	double howManyDroop, howManyIncomplete, howManyAdvance;
	for(i=0;i<num;i++) {
		if (extSim->states[i].state < numBases) {
			if (dnaTemplate[extSim->states[i].state].base == flowOrder[flowNum%numFlowsPerCycle]) { // this one extends
				howManyDroop = extSim->states[i].strands * dr; // MGD - we are not scaling droop with HP now per JD.  * dnaTemplate[extSim->states[i].state].count;
				if (howManyDroop < 1.0)
					howManyDroop = 0.0;
				howManyIncomplete = (extSim->states[i].strands-howManyDroop) * ie;
				if (howManyIncomplete < 1.0)
					howManyIncomplete = 0.0;
				howManyAdvance = extSim->states[i].strands - howManyIncomplete - howManyDroop;
				assert(dnaTemplate[extSim->states[i].state].count < nHpSignal);
				signal += howManyAdvance * hpSignal[dnaTemplate[extSim->states[i].state].count] * sigMult;
				if (howManyIncomplete > 0.0)
					AddState(extSim, howManyIncomplete, extSim->states[i].state);
				extSim->states[i].strands -= (howManyIncomplete+howManyDroop);
				extSim->states[i].state++;
			}

			if (applyCAF && (extSim->states[i].state < numBases)) { // Normally, since we have already 'called' bases, we can model carry-forward, but not past where we already have calls, so its then questionable where to put this signal
				int base_carry = flowOrder[(flowNum-1+numFlowsPerCycle)%numFlowsPerCycle]; // the last reagent - note: I'm assuming that this is the only reagent, the one from two ago would be 0.02 * 0.02 in quantity (roughly) so its negligable (1/2500th of our signal)
				if (dnaTemplate[extSim->states[i].state].base == base_carry) { // this one extends due to carry forward
					double howMany = extSim->states[i].strands * caf;
					howManyDroop = howMany * dr; // MGD - no HP droop scaling per JD  * dnaTemplate[extSim->states[i].state].count;
					howMany -= howManyDroop;
					if (howMany >= 1.0) {
						AddState(extSim, howMany, extSim->states[i].state+1);
						extSim->states[i].strands -= (howMany+howManyDroop);
						assert(dnaTemplate[extSim->states[i].state].count < nHpSignal);
						signal += howMany * hpSignal[dnaTemplate[extSim->states[i].state].count] * sigMult;

#ifdef CARRY_FORWARD_2
						if (extSim->states[i].state+1 < numBases) {
							if (dnaTemplate[extSim->states[i].state+1].base == flowOrder[flowNum%numFlowsPerCycle]) { // so now the portion that carried forward can cause the original nuke to extend it too!
								double howMany2 = howMany * ie;
								if (howMany2 < 1.0) // here, when we are down to 1/10000th of our signal, we just drop it
									howMany2 = 0.0;
								double howMany3 = howMany - howMany2;
								AddState(extSim, howMany3, extSim->states[i].state+2);
								extSim->states[i].strands -= howMany3; // not sure if its really strands from this slice of the pie that we should be removing?
								assert(dnaTemplate[extSim->states[i].state+1].count < nHpSignal);
								signal += howMany3 * hpSignal[dnaTemplate[extSim->states[i].state+1].count] * sigMult;
							}
						}
#endif /* CARRY_FORWARD_2 */
					}
				}
			}
		}
	}


	signal /= strandCount;
	// printf("Measured signal is: %.2lf\n", signal);
	return signal;
}
#endif /* NOT_USED */

double CafieSolver::TestReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool applyCAF)
{
	double hpSignal[MAX_MER];
	double sigMult=1.0;

	for(int i=0; i < MAX_MER; i++)
		hpSignal[i] = i;
	
	return(ApplyReagentFast(extSim,dnaTemplate,numBases,flowNum,ie,caf,dr,applyCAF,true,hpSignal,MAX_MER,sigMult));
}

double CafieSolver::TestReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool applyCAF, double *hpSignal, int nHpSignal, double sigMult)
{	//Unused parameter generates compiler warning, so...
	if (nHpSignal) {};
	return ApplyReagentFast(extSim,dnaTemplate,numBases,flowNum,ie,caf,dr,applyCAF,true,hpSignal,MAX_MER,sigMult);
}


void CafieSolver::FreeExtensionSim(ExtensionSim *extSim)
{
#ifndef NO_MALLOC
	if (extSim->states)
		free(extSim->states);
	extSim->states = NULL;
#endif /* NO_MALLOC */
	extSim->numStates = 0;
}

void CafieSolver::InitExtensionSim(ExtensionSim *extSim)
{
	FreeExtensionSim(extSim);
	AddState(extSim, strandCount, 0);
}

void CafieSolver::CopyExtensionSim(ExtensionSim *extSim, ExtensionSim *extSimCopy)
{
#ifndef NO_MALLOC
	extSimCopy->states = (ExtensionState *)malloc(extSim->numStates * sizeof(ExtensionState));
#endif /* NO_MALLOC */
	memcpy(extSimCopy->states, extSim->states, extSim->numStates * sizeof(ExtensionState));
	extSimCopy->numStates = extSim->numStates;
}

void CafieSolver::SetFlowOrder(char *flowChars)
{
	if (flowOrder)
		free(flowOrder);
	flowOrder = strdup(flowChars);
	numFlowsPerCycle = strlen(flowOrder);

	if (flowOrderIndex)
		delete [] flowOrderIndex;
	flowOrderIndex = new int[numFlowsPerCycle];
	SetFlowOrderIndex(flowOrderIndex,flowOrder,numFlowsPerCycle);
	DistToBaseInit();
}

int CafieSolver::GetNuc(int flow)
{
	// nuc's by definition are 0=T, 1=A, 2=C, 3=G - this is the default flow order on the PGM, so we wanted to make sure it makes sense for our standard
	return flowOrderIndex[flow%numFlowsPerCycle];
}

int CafieSolver::DotTest(int curFlow, int *predictedExtension)
{
	// algorithm here is to look back until we find all 4 nucs,
	// if found and all flows except the oldest are 0-mers, then we have a dot case
	int flow;
	unsigned long nucsFound = 0;
	for(flow=curFlow;flow>=0;flow--) {
		// see if we have all 4 nucs, if we did we also did not exit earlier due to non 0-mer check below, we we found a dot
		nucsFound |= (1 << flowOrderIndex[flow%numFlowsPerCycle]);
		if (nucsFound == 0xf)
			return (curFlow-flow-1);
		// see if current flow is not a 0-mer, if its not we can safely bail
		if ((flow < curFlow) && (predictedExtension[flow] > 0)) // tricky stuff here, the predictedExtension array does not yet contain the prediction for the current nuc on this pass through the algorithm, so we don't check it.  Further, its assumed that it would be zero since thats a condition we use when calling this method
			return 0;
	}
	return 0;
}

void CafieSolver::SetTestSequence(char *testSequence)
{
	strcpy(currentTestSequence, testSequence);
	memset(dnaTemplate, 0, sizeof(dnaTemplate));

	// convert a dna template from the source template string into our internal structure
	testLen = 0;
	numTemplateBases = 0;
	int len = (int)strlen(testSequence);
	int i;
	char lastChar = testSequence[0];
	for(i=0;i<len;i++) {
		if (lastChar != testSequence[i])
			testLen++;
		dnaTemplate[testLen].base = testSequence[i];
		dnaTemplate[testLen].count++;
		lastChar = testSequence[i];
	}
	testLen++; // now this indicates the number of bases in our test sequence (each homopolymer stretch counts as one, no matter the length)
}

void CafieSolver::SimulateCAFIE(double* predicted, const char* seq, const char* flowOrder, double cf, double ie, double dr, int nflows)
{
	double hpSignal[MAX_MER];
	double sigMult=1.0;

	for(int i=0; i < MAX_MER; i++)
		hpSignal[i] = i;
	
	SimulateCAFIE(predicted,seq,flowOrder,cf,ie,dr,nflows,hpSignal,MAX_MER,sigMult);
}

void CafieSolver::SimulateCAFIE(double* predicted, const char* seq, const char* flowOrder, double cf, double ie, double dr, int nflows, double *hpSignal, int nHpSignal, double sigMult)
{
	SetFlowOrder(const_cast<char*>(flowOrder));
	SetTestSequence(const_cast<char*>(seq));
	
	InitExtensionSim(&extSim);
	for(int i=0; i<nflows; i++){
		predicted[i] = ApplyReagent(&extSim, dnaTemplate, testLen, i, ie, cf, dr, false, hpSignal, nHpSignal, sigMult);
		CompactStates(&extSim);
	}

/*
	double drMult = 1.0;
	for(int i=0; i<nflows; i++){
		predicted[i] *= drMult;
		drMult *= (1.0-dr);
	}
*/
}

void CafieSolver::Model(double *ideal, int numIdealFlows, float *predicted, int numPredictedFlows, double cfModeled, double ieModeled, double drModeled)
{
	// build our DNA template from the ideal vector & flow order
	testLen = 0;
	int i;
	for(i=0;i<numIdealFlows;i++) {
		if (ideal[i] > 0.0) {
			dnaTemplate[testLen].count = (int)(ideal[i]+0.5);
			if(dnaTemplate[testLen].count >= MAX_MER)
				dnaTemplate[testLen].count = MAX_MER-1;
			dnaTemplate[testLen].base = flowOrder[i%numFlowsPerCycle];
			testLen++;
		}
	}

	// add phase errors to the ideal vector to produce the non-droop predictions, and apply droop
	InitExtensionSim(&extSim);
	// double localDroopMult = 1.0;
	// double localPredicted;
	for(i=0;i<numPredictedFlows;i++) {
		// get the predicted measurement for flow i
		// localPredicted = ApplyReagent(&extSim, dnaTemplate, testLen, i, ieModeled, cfModeled, drModeled, false);
		predicted[i] = ApplyReagent(&extSim, dnaTemplate, testLen, i, ieModeled, cfModeled, drModeled, false);
		// predicted[i] = ApplyReagentFast(&extSim, dnaTemplate, testLen, i, ieModeled, cfModeled, drModeled, (i>0));

		// add droop to the predicted value
		// localPredicted *= localDroopMult;
		// predicted[i] = (float)localPredicted; // for convenience to the levmar caller, I'm passing the predicted vals in as floats
		// localDroopMult *= (1.0 - drModeled);

		// keep the extension sim compact and efficient
		CompactStates(&extSim);
// debug
/*
printf("Flow %d\n", i);
int state;
for(state=0;state<extSim.numStates;state++) {
	printf("%d:%.5lf ", extSim.states[state].state, (double)extSim.states[state].strands/(double)strandCount);
}
printf("\n");
*/
	}
}

void CafieSolver::ModelIdeal(double *ideal, int numIdealFlows, double *predicted, double *predictedDelta, int numPredictedFlows, double *measured, int numMeasured, double cfModeled, double ieModeled, double drModeled)
{	//Unused parameter generates compiler warning, so...
	if (numMeasured) {};
	
	DNATemplate dnaTemplateGuess[MAX_TEMPLATE_SIZE];
	// DNATemplate dnaTemplateTest[MAX_TEMPLATE_SIZE];
	int dnaTemplateGuessLen = 0;
	int i;
	int flow;

	// zero-out the new ideal flows
	for(i=numIdealFlows;i<numPredictedFlows;i++)
		ideal[i] = 0.0;
	int numValidIdeal = numIdealFlows; // start with all prior knowledge, this will increase as we start to discover our new ideal vector portion

	InitExtensionSim(&extSim);
	// ExtensionSim testSim;

	memset(dnaTemplateGuess, 0, sizeof(DNATemplate) * MAX_TEMPLATE_SIZE);
	for(i=0;i<numIdealFlows;i++) {
		double val = ideal[i];
		if (val > 0.0) {
			dnaTemplateGuess[dnaTemplateGuessLen].count = (int)(val+0.5);
			dnaTemplateGuess[dnaTemplateGuessLen].base = flowOrder[i%numFlowsPerCycle];
			dnaTemplateGuessLen++;
		}
	}
	int dnaTemplateGuessCur = 0;

	for(flow=0;flow<numPredictedFlows;flow++) {

		int bestMer = 0;

		// generate a test template
		if (dnaTemplateGuessCur >= dnaTemplateGuessLen) { // new base tried at the end
			dnaTemplateGuess[dnaTemplateGuessCur].base = flowOrder[flow%numFlowsPerCycle];
			dnaTemplateGuess[dnaTemplateGuessCur].count = 0;
			dnaTemplateGuessLen++;
		} else { // new base either inserted or overriding existing
			if (dnaTemplateGuess[dnaTemplateGuessCur].base == flowOrder[flow%numFlowsPerCycle]) { // just re-trying existing
				// nothing to do here
			} else {
				// move existing sequence out to make room for new guess
				for(i=dnaTemplateGuessLen;i>dnaTemplateGuessCur;i--) {
					dnaTemplateGuess[i].base = dnaTemplateGuess[i-1].base;
					dnaTemplateGuess[i].count = dnaTemplateGuess[i-1].count;
				}

				// insert the new guess
				dnaTemplateGuess[dnaTemplateGuessCur].base = flowOrder[flow%numFlowsPerCycle];
				dnaTemplateGuess[dnaTemplateGuessCur].count = 0;

				// and update extension sim state pointers (since we added a base to our guess, any state indexes in the extension sim must now be corrected)
				for(i=0;i<extSim.numStates;i++) {
					if (extSim.states[i].state > dnaTemplateGuessCur)
						extSim.states[i].state++;
				}

				dnaTemplateGuessLen++;
			}
		}

/*
printf("Cur pos start: %d\n", dnaTemplateGuessCur);
printf("Current test seq: ");
for(i=0;i<dnaTemplateGuessLen;i++) {
	if (i == dnaTemplateGuessCur)
		printf("%c%d", tolower(dnaTemplateGuess[i].base), dnaTemplateGuess[i].count);
	else {
		for(int j=0;j<dnaTemplateGuess[i].count;j++)
			printf("%c", dnaTemplateGuess[i].base);
	}
}
printf("\n");

if ((int)(ideal[flow]+0.5) != dnaTemplateGuess[dnaTemplateGuessCur].count) {
	printf("Possible out of sync here?\n");
	// ideal[flow] = dnaTemplateGuess[dnaTemplateGuessCur].count;
}
*/
		double sig[MAX_MER];
		// int flowToModel = numValidIdeal;
		// if ((flow+1) > flowToModel)
			// flowToModel = flow+1;

		// int dnaTemplateGuessLen2 = dnaTemplateGuessLen;

		int mer;
		for(mer=0;mer<3;mer++) {
			dnaTemplateGuess[dnaTemplateGuessCur].count = mer;
			sig[mer] = ApplyReagentFast(&extSim, dnaTemplateGuess, dnaTemplateGuessLen, flow, ieModeled, cfModeled, drModeled, (flow>0), true);
		}

// printf("signals: %.5lf %.5lf %.5lf  measured: %.5lf\n", sig[0], sig[1], sig[2], measured[flow]);

		predictedDelta[flow] = sig[2] - sig[1];
		for(mer=3;mer<MAX_MER;mer++) {
			sig[mer] = sig[2] + predictedDelta[flow] * (mer-2);
		}

		double maxDelta = 0.0;
		double delta;
		for(mer=0;mer<MAX_MER;mer++) {
			delta = sig[mer] - measured[flow];
			if (delta < 0) delta = -delta;
			if (delta < maxDelta || mer == 0) {
// printf("Flow: %d best so far: %d with delta: %.5lf\n", flow, mer, delta);
				maxDelta = delta;
				bestMer = mer;
				predicted[flow] = sig[mer];
			}
		}
		if (bestMer == (MAX_MER-1))
			bestMer = 0;

		// fix dots
		bool fullRegenerate = false;
bool fixdots = true;
		if (fixdots && flow > 3) {
			if (bestMer == 0 && ideal[flow-1] == 0 && ideal[flow-2] == 0) {
				int flowToCorrect = 0;
				if (predictedDelta[flow-2] > predictedDelta[flow-1])
					flowToCorrect = flow-2;
				else
					flowToCorrect = flow-1;
				if (predictedDelta[flow] > predictedDelta[flowToCorrect])
					flowToCorrect = flow;
				// flowToCorrect = flow; // old way was to make the most recent flow a 1-mer
				bestMer = 1;
				fullRegenerate = true;
				flow = flowToCorrect; // we want to pick off where we left off as if we had called the flowToCorrect as a 1-mer in the first place
// printf("Dot\n");
			}
		}

		// keep track of the ideal vector (this is in flow space)
		int curMer = ideal[flow];
		ideal[flow] = bestMer;
		if ((flow+1) > numValidIdeal)
			numValidIdeal = flow+1;
// printf("cur: %d  new: %d\n", curMer, bestMer);

		// update our model
// fullRegenerate = true;
bool changed = false;
		if (fullRegenerate) {
			dnaTemplateGuessLen = 0;
			dnaTemplateGuessCur = 0;
			for(i=0;i<numValidIdeal;i++) {
				double val = ideal[i];
				if (val > 0.0) {
					dnaTemplateGuess[dnaTemplateGuessLen].count = (int)(val+0.5);
					dnaTemplateGuess[dnaTemplateGuessLen].base = flowOrder[i%numFlowsPerCycle];
					dnaTemplateGuessLen++;
					if (i <= flow) {
						dnaTemplateGuessCur++;
					}
				}
			}
			changed = true;
		} else {
			if (curMer != bestMer) {
				if (bestMer == 0) {
// printf("Delete (guessCur=%d  guessLen:%d)\n", dnaTemplateGuessCur, dnaTemplateGuessLen);
					for(i=dnaTemplateGuessCur;i<dnaTemplateGuessLen-1;i++) {
						dnaTemplateGuess[i].base = dnaTemplateGuess[i+1].base;
						dnaTemplateGuess[i].count = dnaTemplateGuess[i+1].count;
					}
					dnaTemplateGuessLen--; // we used to have a base, but it turned out to be a 0-mer
				} else {
// printf("Insert\n");
					dnaTemplateGuess[dnaTemplateGuessCur].count = bestMer;
					dnaTemplateGuessCur++; // we now (or still do) have a base at this position, advance
					if (dnaTemplateGuessCur < dnaTemplateGuessLen)
						changed = true;
				}
				// changed = true;
			} else { // curMer == bestMer
				dnaTemplateGuess[dnaTemplateGuessCur].count = bestMer;
				if (curMer == 0) {
					for(i=dnaTemplateGuessCur;i<dnaTemplateGuessLen-1;i++) {
						dnaTemplateGuess[i].base = dnaTemplateGuess[i+1].base;
						dnaTemplateGuess[i].count = dnaTemplateGuess[i+1].count;
					}
					dnaTemplateGuessLen--; // we tried to add a base, but it turned out to be a 0-mer
					changed = true;
				} else
					dnaTemplateGuessCur++; // this base didn't change and is not a 0-mer, so advance to the next position to try
			}
		}

		if (changed) {
			InitExtensionSim(&extSim);
			for(i=0;i<flow;i++) {
				// ApplyReagent(&extSim, dnaTemplateGuess, dnaTemplateGuessLen, i, ieModeled, cfModeled, drModeled, false);
				ApplyReagentFast(&extSim, dnaTemplateGuess, dnaTemplateGuessLen, i, ieModeled, cfModeled, drModeled, (i>0), false);
				if (i%2 == 0)
					CompactStates(&extSim);
			}
		}

		ApplyReagentFast(&extSim, dnaTemplateGuess, dnaTemplateGuessLen, flow, ieModeled, cfModeled, drModeled, (flow>0), false);
		CompactStates(&extSim);
/*
for(i=0;i<extSim.numStates;i++) {
	printf("%d:%.5lf ", extSim.states[i].state, extSim.states[i].strands);
}
printf("\n");
printf("Cur pos end: %d\n", dnaTemplateGuessCur);
printf("\n");
*/
	}
}

int CafieSolver::SolveIdeal(double droopEstimate, int droopMode, bool reNormalize, ChipIdEnum phredScoreVersion, bool scaledSolve, unsigned long filterTests)
{
	int numFlowsToFit; // number of flows we are currently fitting to
	int numFlowsToUse = 40; // number of flows we can use for the fit - so it is always equal to or greater than numFlowsToFit - to give us carry-forward correction ability.  Its also the number of flows we 'basecall'
	double cfUseRatio = 0.85; // the percentage of flows we can accuratly(safely) call due to carry-forward (we need the pre-cog flows)
	int numFlowsInc = 25; // how many flows to advance by with each increment
	int numFlowsInc2 = 0; // increments numFlowsInc each time
	int flowLimit; // stop fitting cf/ie/dr after this many flows
	flowLimit = numFlowsToUse + 2 * numFlowsInc;
	// flowLimit = (int)(numFlows * 0.75 + 0.5); // MGD - this hurt performance and took forever to run!
	double curCF = meanCaf; // 0.01; // really good initial guess based on pre-cog - MGD - will need to update as we get better data!
	double curIE =  meanIe; // 0.008;
	double curDR = meanDr; // 0.0015;
/*
curCF = 0.0;
curIE = 0.0;
meanDr = 0.0;
*/


	double ideal[numFlows];
	//double predicted[numFlows];
	double predictedDelta[numFlows];
	float residual[numFlows];
	int flow;
	int ret = 0; // so far so good
	// bool avgResidualFilterApplied = false; // only want to attemt this filter once, but can't predict what the flow count will be, so its a range

	// default params (for droopMode = 0)
	CfiedrParams minParams, maxParams;
	minParams.cf = 0.0; //-0.1;
	minParams.ie = 0.0; //-0.1;
	minParams.dr = -0.1;
	maxParams.cf = 0.1;
	maxParams.ie = 0.1;
	maxParams.dr = 0.1;

	if (droopMode == 1) { // estimate droop from the read itself rather than fitting as part of cf/ie/dr
		double estimatedLibraryDroop = EstimateLibraryDroop(measured, numFlows);
		// printf("Estimated library droop: %.5lf\n", estimatedLibraryDroop);
		if (estimatedLibraryDroop >= 0.0 && estimatedLibraryDroop < 0.01) {
			minParams.dr = estimatedLibraryDroop;
			maxParams.dr = estimatedLibraryDroop;
			curDR = estimatedLibraryDroop;
		}
	} else if (droopMode == 2) { // use the droop the caller passed in
		minParams.dr = droopEstimate;
		maxParams.dr = droopEstimate;
		curDR = droopEstimate;
	}

	memset(ideal, 0, sizeof(double) * numFlows);

	// bootstrap system with first few flows & no cf/ie/dr corrections and no initial ideal vector
	// MGD - as an alternative, we could first apply a best guess for cf/ie/dr instead
	int curIdealSize = 0;
	numFlowsToFit = (int)(numFlowsToUse * cfUseRatio + 0.5);
	ModelIdeal(ideal, curIdealSize, predicted, predictedDelta, numFlowsToUse, measured, numFlows, curCF, curIE, curDR);

	// re-normalize
	if (reNormalize) {
		ModelIdeal(ideal, numFlowsToUse, predicted, predictedDelta, numFlowsToUse, measured, numFlows, curCF, curIE, curDR);
		int idealKeyVec[numFlowsToUse];
		for(flow=0;flow<numFlowsToUse;flow++) {
			if (flow < 12)
				idealKeyVec[flow] = -1;
			else
				idealKeyVec[flow] = (int)(ideal[flow]+0.5);
		}
		bool perNuc = false;
		SetMeasured(numFlows, origMeasured);
		Normalize(idealKeyVec, (int)(numFlowsToUse*0.7+0.5), 0.0, false, perNuc);
		// Normalize(idealKeyVec, 8, 0.0, false, perNuc); // intent here was to use the G where possible

		// and re-model with updated measurements
		ModelIdeal(ideal, curIdealSize, predicted, predictedDelta, numFlowsToUse, measured, numFlows, curCF, curIE, curDR);
	}

/*
printf("Ideal1: ");
for(int i=0;i<numFlowsToUse;i++)
	printf("%d ", (int)ideal[i]);
printf("\n");
*/
	// after ModelIdeal, we have numFlowsToUse values in our ideal vector

	while (numFlowsToUse < numFlows) {
		if (numFlowsToUse <= flowLimit) {
			// fit to the flows and ideal vector we have so far and refine the cf/ie/dr
			CfiedrFit cfiedrFit(numFlowsToUse, numFlowsToFit, this, true);
			// CfiedrFit cfiedrFit(numFlowsToUse, numFlowsToFit, this, true);
			cfiedrFit.SetParamMin(minParams);
			cfiedrFit.SetParamMax(maxParams);
			cfiedrFit.params.cf = curCF; // 0.0; //maxParams.cf/2.0;
			cfiedrFit.params.ie = curIE; // 0.0; //maxParams.ie/2.0;
			cfiedrFit.params.dr = curDR; // 0.0; //maxParams.dr/2.0;
			cfiedrFit.Init(ideal);
#ifndef CFIEDR_FIT_V1
			cfiedrFit.SetLambdaStart(1.0);
			cfiedrFit.SetLambdaThreshold(1E10);
#endif /* not CFIEDR_FIT_V1 */
			/* int iter = */cfiedrFit.Fit(10);
			/* double residual = */cfiedrFit.GetResidual();

			// get our new cf, ie, dr params found from the fit
			curCF = cfiedrFit.params.cf;
			curIE = cfiedrFit.params.ie;
			curDR = cfiedrFit.params.dr;

/*
printf("use: %d  fit: %d  cf/ie/dr: %.5lf/%.5lf/%.5lf iter: %d  res: %.5lf\n",
	numFlowsToUse, numFlowsToFit, curCF, curIE, curDR, iter, residual);
*/

		} else {
			// numFlowsInc = numFlows; // forces numFlowsToUse to just be all flows
			numFlowsInc = 50;
		}

		// re-normalize based on our latest corrected data
		if (scaledSolve) {
			for(flow=0;flow<numFlowsToUse;flow++)
				predictedExtension[flow] = (int)ideal[flow];
			// ResidualScale();

			bool nucSpecific=false;
			int minFlow = 12;
			int maxFlow = (numFlowsToUse > 50 ? 50 : numFlowsToUse);
			int nWeight=MAX_MER;
			double weight[MAX_MER];
			for(int i=0; i<MAX_MER; i++)
				weight[i] = 0;
			weight[1] = 1;
			weight[2] = 0.8;
			weight[3] = 0.64;
			weight[4] = 0.512;
			double minObsValue=MIN_NORMALIZED_SIGNAL_VALUE;

			ResidualScale(nucSpecific, minFlow, maxFlow-1, weight, nWeight, minObsValue);
		}

		// extend our basecall length
		curIdealSize = numFlowsToUse;
		numFlowsToUse += numFlowsInc;
		numFlowsInc += numFlowsInc2;
		if (numFlowsToUse > numFlows)
			numFlowsToUse = numFlows;
		numFlowsToFit = (int)(numFlowsToUse * cfUseRatio + 0.5);

		// update corrected flows based on new cf/ie/dr params
		ModelIdeal(ideal, curIdealSize, predicted, predictedDelta, numFlowsToUse, measured, numFlows, curCF, curIE, curDR);

		// filters

		// HP filter - look for avg base calls > 4 over a 5 flow window, if found, then stop correcting, this read is not going to work
/* - OK, this one didn't seem to work
		if (numFlowsToUse <= 180) { // after 180 flows we are in the clear
			for(flow=0;flow<numFlowsToUse-5;flow++) {
				int flow2;
				double avg = 0.0;
				for(flow2=flow;flow2<(flow+5);flow2++) {
					avg += ideal[flow2];
				}
				if (avg*0.2 >= 4.0) {
					ret = 1;
					break; // out of the while loop
				}
			}
		}
*/

		// Avg residual filter
/* - OK, this didn't help either?  One plot shows this should have helped determine what reads to keep
		if (numFlowsToUse >= 80 && !avgResidualFilterApplied) {
			avgResidualFilterApplied = true;
			double avgResidual = 0.0;
			for(flow=0;flow<numFlowsToUse;flow++) {
				avgResidual += fabs(predicted[flow] - measured[flow]);
			}
			avgResidual /= numFlowsToUse;
			if (avgResidual > 0.25) {
				ret = 1;
				break; // out of the while loop
			}
		}
*/

/*
printf("Ideal: ");
for(int i=0;i<numFlowsToUse;i++)
	printf("%d ", (int)ideal[i]);
printf("\n");
*/
	}

	// re-normalize based on our latest corrected data
/*
	if (scaledSolve) {
		for(flow=0;flow<numFlows;flow++)
			predictedExtension[flow] = (int)ideal[flow];
		ResidualScale();
	}
*/

	// final pass to improve things
	// curIdealSize = numFlowsToUse;
	// ModelIdeal(ideal, curIdealSize, predicted, predictedDelta, numFlowsToUse, measured, numFlows, curCF, curIE, curDR);

	// calculate the corrected ionogram
	double scaledError;
	numCalls = 0;
	for(flow=0;flow<numFlows;flow++) {
		residual[flow] = predicted[flow] - measured[flow];
		scaledError = residual[flow]/predictedDelta[flow];
		
		corrected[flow] = ideal[flow] + scaledError;
		if (corrected[flow] >= MAX_MER)
			corrected[flow] = MAX_MER;
		predictedExtension[flow] = (int)ideal[flow];
		numCalls += predictedExtension[flow];
	}


/*
	printf("\nBases called via thresh: ");
	for(flow=0;flow<numFlows;flow++) {
		int base = (int)(corrected[flow]+0.5);
		if (base >= MAX_MER)
			base = MAX_MER-1;
		while (base > 0) {
			printf("%c", flowOrder[flow % numFlowsPerCycle]);
			base--;
		}
	}
	printf("\n");

	printf("\nBases called via predic: ");
	for(flow=0;flow<numFlows;flow++) {
		int base = predictedExtension[flow];
		if (base >= MAX_MER)
			base = MAX_MER-1;
		while (base > 0) {
			printf("%c", flowOrder[flow % numFlowsPerCycle]);
			base--;
		}
	}
	printf("\n");
*/

	meanCaf = curCF;
	meanIe = curIE;
	meanDr = curDR;

	return ret;
}

double CafieSolver::FindBestCAFIE(double *measuredVals, double* predictedVals, int numVals)
{
	// this version is modified to extract additional data, without getting in the way 
	// of the version below that's called by Analysis.cpp
	
	// brute-force search through the CAF & IE space for the best fit to the measured data,
	// predictions are calcualted from the test sequence

	int    cfNum  = 51;
	int    ieNum  = 51;
	double cfLow  = 0.0;
	double ieLow  = 0.0;
	double cfHigh = 0.05;
	double ieHigh = 0.05;
	double cfInc  = (cfHigh-cfLow)/(double)(cfNum-1.0);
	double ieInc  = (ieHigh-ieLow)/(double)(ieNum-1.0);

	double** predictedValue = (double**)malloc(sizeof(double *) * cfNum * ieNum);

	int k = 0;
	for(int icf=0; icf<cfNum; ++icf){
		double caf = cfLow + icf*cfInc;
		for(int iie=0; iie<ieNum; iie++) {
			double ie = ieLow + iie*ieInc;
			predictedValue[k] = (double *)malloc(sizeof(double) * numVals);
			// initialize the simulation states
			InitExtensionSim(&extSim);
			for(int i=0;i<numVals;i++) {
				predictedValue[k][i] = ApplyReagent(&extSim, dnaTemplate, testLen, i, ie, caf, 0.0, false);
				CompactStates(&extSim);
			}
			k++;
		}
	}

	// search the CAFIE space, for each pair, calculate the err, store CAFIE pair for min err
	double bestErr = 99999999999999.0;

	// clamp to our max handled template length
	assert(numVals <= MAX_TEMPLATE_SIZE);

	double droopLow  = 0.0;
	double droopHigh = 0.01;
	double droopInc  = 0.00025;

	double* correctedVals = new double[numVals];
	for(double dr=droopLow; dr<=droopHigh; dr+=droopInc) {
		// apply droop
		double droopMult = 1.0;
		for(int i=0;i<numVals;i++) {
			correctedVals[i] = measuredVals[i] * droopMult;
			droopMult /= (1.0-dr);
		}

		k = 0;
		for(int icf=0; icf<cfNum; icf++) {
			for(int iie=0; iie<ieNum; iie++) {
				double err = 0.0;
				for(int i=0;i<numVals;i++) {
				//for(int i=0;i<40;i++) {
					double errsq = (predictedValue[k][i] - correctedVals[i]);
					//errsq = (predictedValue[k][i] - round(predictedValue[k][i]));
					errsq = errsq * errsq;

					err += errsq;
				}

				// see if this is the 'best' error
				if (err < bestErr) {
					meanCaf = cfLow + icf*cfInc;;
					meanIe = ieLow + iie*ieInc;
					meanDr = 1.0-dr;
					bestErr = err;
					for(int i=0;i<numVals;i++)
						predicted[i] = predictedValue[k][i]; // store result of best prediction CAFIE pair thus far
				}

				k++;
			}
		}
	}

    delete [] correctedVals;

	for(int flow=0; flow<numVals; ++flow)
		predictedVals[flow] = predicted[flow];

	return bestErr;
}

double CafieSolver::FindBestCAFIE(double *measuredVals, int numVals, bool useDroopEst, double droopEst)
{
	double hpSignal[MAX_MER];
	double sigMult=1.0;

	for(int i=0; i < MAX_MER; i++)
		hpSignal[i] = i;
	
	return(FindBestCAFIE(measuredVals,numVals,useDroopEst,droopEst,hpSignal,MAX_MER,sigMult));
}

double CafieSolver::FindBestCAFIE(double *measuredVals, int numVals, bool useDroopEst, double droopEst, double *hpSignal, int nHpSignal, double sigMult)
{
	// brute-force search through the CAF & IE space for the best fit to the measured data,
	// predictions are calcualted from the test sequence

	// Make sure the hpSignal array is long enough
	assert(nHpSignal >= MAX_MER);

	// first, see if an pre-calculated Ionogram grid already exists, if now, generate now
	double caf, ie, dr;
	int icf, iie;
	int i;
	int k;
	int iGrid = -1;
	for(i=0;i<numIonogramGrids;i++) {
		if (strcmp(currentTestSequence, ionogramGrid[i].seq) == 0) {
			bool hpSignalEqual=true;
			for(int j=0; j<MAX_MER; j++) {
				if(abs(hpSignal[j]-ionogramGrid[i].hpSignal[j]) > 1e-8) {
					hpSignalEqual=false;
					break;
				}
			}
			if(hpSignalEqual & (fabs(sigMult-ionogramGrid[i].sigMult) < 1e-8))
				iGrid = i;
			if(iGrid > -1)
				break;
		}
	}

	if (iGrid == -1) {
		fprintf(stdout, "*** Generating Ionogram grid for %s\n", currentTestSequence);
		strcpy(ionogramGrid[numIonogramGrids].seq, currentTestSequence);
		ionogramGrid[numIonogramGrids].cfNum = 51;
		ionogramGrid[numIonogramGrids].ieNum = 51;
		ionogramGrid[numIonogramGrids].cfLow = 0.0;
		ionogramGrid[numIonogramGrids].ieLow = 0.0;
		ionogramGrid[numIonogramGrids].cfHigh = 0.05;
		ionogramGrid[numIonogramGrids].ieHigh = 0.05;
		ionogramGrid[numIonogramGrids].cfInc = (ionogramGrid[numIonogramGrids].cfHigh-ionogramGrid[numIonogramGrids].cfLow)/(double)(ionogramGrid[numIonogramGrids].cfNum-1.0);
		ionogramGrid[numIonogramGrids].ieInc = (ionogramGrid[numIonogramGrids].ieHigh-ionogramGrid[numIonogramGrids].ieLow)/(double)(ionogramGrid[numIonogramGrids].ieNum-1.0);
		for(int j=0; j<MAX_MER; j++)
			ionogramGrid[numIonogramGrids].hpSignal[j] = hpSignal[j];
		ionogramGrid[numIonogramGrids].sigMult = sigMult;
		ionogramGrid[numIonogramGrids].numFlows = numVals;
		ionogramGrid[numIonogramGrids].predictedValue = (double **)malloc(sizeof(double *) * ionogramGrid[numIonogramGrids].cfNum * ionogramGrid[numIonogramGrids].ieNum);

		// now generate the grid of Ionograms
		k = 0;
		for(icf=0;icf<ionogramGrid[numIonogramGrids].cfNum;icf++) {
			caf = ionogramGrid[numIonogramGrids].cfLow + icf*ionogramGrid[numIonogramGrids].cfInc;
			for(iie=0;iie<ionogramGrid[numIonogramGrids].ieNum;iie++) {
				ie = ionogramGrid[numIonogramGrids].ieLow + iie*ionogramGrid[numIonogramGrids].ieInc;
				ionogramGrid[numIonogramGrids].predictedValue[k] = (double *)malloc(sizeof(double) * numVals);
				// initialize the simulation states
				InitExtensionSim(&extSim);
				for(i=0;i<numVals;i++) {
					ionogramGrid[numIonogramGrids].predictedValue[k][i] = ApplyReagent(&extSim, dnaTemplate, testLen, i, ie, caf, 0.0, false, hpSignal, nHpSignal, sigMult);
					CompactStates(&extSim);
				}
				k++;
			}
		}
		iGrid = numIonogramGrids;
		numIonogramGrids++;
	}

	// search the CAFIE space, for each pair, calculate the err, store CAFIE pair for min err
	double err, errsq;
	double bestErr = 99999999999999.0;
	double	droopLow, droopHigh, droopInc;

	// clamp to our max handled template length
	if (numVals > MAX_TEMPLATE_SIZE)
		numVals = MAX_TEMPLATE_SIZE;

/*
	// calculate the droop ratio
	int droopIncorporations = 0;
	int flow;
	for(flow=0;flow<ionogramGrid[iGrid].numFlows;flow++) {
		// MGD - this version good for n-mer incorporations each adding to droop - droopIncorporations += (int)(ionogramGrid[iGrid].predictedValue[0][flow]+0.5);
		droopIncorporations += (ionogramGrid[iGrid].predictedValue[0][flow] >= 0.5 ? 1 : 0);
	}
	double droopRatio = droopIncorporations/(double)ionogramGrid[iGrid].numFlows;
*/
	// set droop search range, or use user input
	if (useDroopEst) {
		droopLow = droopEst;
		droopHigh = droopEst;
		droopInc = 1.0; // just had to be bigger than droopEst so we only loop once
	} else {
		droopLow = 0.0;
		droopHigh = 0.01;
		droopInc = 0.00025;
	}

	for(dr=droopLow;dr<=droopHigh;dr+=droopInc) {
		k = 0;
		for(icf=0;icf<ionogramGrid[iGrid].cfNum;icf++) {
			for(iie=0;iie<ionogramGrid[iGrid].ieNum;iie++) {
				err = 0.0;
				int errCount = 0;
				double droopMult = 1.0; // incorporation-based droom model multiplier - assumes droop is roughly separable from phase
				for(i=0;i<numVals;i++) {
					// un-drooped pre-calc'ed predictions adjusted for incorporation-based droop model, then delta from measured calculated
					errsq = (ionogramGrid[iGrid].predictedValue[k][i]*droopMult - measuredVals[i]);
					errsq = errsq * errsq;

					if (ionogramGrid[iGrid].predictedValue[0][i] < 2.5) { // minimize error only for 0,1,2-mers in TF
						err += errsq;
						errCount++;
					}
					droopMult *= (1.0 - dr * (ionogramGrid[iGrid].predictedValue[0][i] >= 0.5 ? 1 : 0)); // apply droop only when incorporating
				}
				if (errCount > 0) {
					err /= errCount;
					err = sqrt(err);
				}

				// see if this is the 'best' error
				if (err < bestErr) {
					meanCaf = ionogramGrid[iGrid].cfLow + icf*ionogramGrid[iGrid].cfInc;;
					meanIe = ionogramGrid[iGrid].ieLow + iie*ionogramGrid[iGrid].ieInc;
					meanDr = dr;
					bestErr = err;
					for(i=0;i<numVals;i++)
						predicted[i] = ionogramGrid[iGrid].predictedValue[k][i]; // store result of best prediction CAFIE pair thus far
				}

				k++;
			}
		}
	}
	
	return bestErr;
}

// Below is the original EstimateLibraryDroop method
//    It may look similar to the new EstimateLibraryDroop but it is kept here
//    for historical reasons.
//double CafieSolver::EstimateLibraryDroop(double *measuredVals, int numMeasured)
//{
//	// simple sliding window approach
//	// window has an initial span, and slides along all measured flows while also increasing its span size
//	// double windowWidth = 9.0;
//	// double windowInc = 0.1;
//	double windowWidth = 1.0; // since we now fit to the data, just use every point, no need to avg - but may want to separate out per nuc eventually
//	double windowInc = 0.0;
//	int start = 16; // start after the key
//	int end = numMeasured/2; // want to end well short of where our shorter templates would stop sequencing
//
//	int numSamples = end - start - windowWidth - (int)(windowInc * (end-start-windowWidth)+0.5);
//	float samples[numSamples];
//
//	int i;
//	for(i=0;i<numSamples;i++) {
//		double avg = 0.0;
//		int j;
//		int width = windowWidth + (int)(windowInc*i+0.5);
//		for(j=0;j<width;j++) {
//			int k = i + j + start;
//			assert(k < numMeasured && "ERROR!  Deveolper bug here - under estimate of numSamples in Library Droop Estimation");
//			avg += measuredVals[k];
//		}
//		samples[i] = avg/width;
//	}
//
//	// function: signal(x) = 0.6 * power((1.0-droop), x)
//	// solve for droop value that minimizes this function at each point x
//	// to do: may want to use the first point as the initial value, rather than 0.6
//	DroopFit droopFit(numSamples);
//	DroopParams drMin, drMax;
//	drMin.dr = 0.0;
//	drMin.base = 0.3;
//	drMax.dr = 0.01;
//	drMax.base = 0.95;
//	droopFit.SetParamMin(drMin);
//	droopFit.SetParamMax(drMax);
//	droopFit.params.dr = 0.0; // initial guess
//	float baseline = (samples[4] + samples[5] + samples[6] + samples[7])*0.25;
//	droopFit.params.base = (baseline > drMax.base ? drMax.base : baseline);
//	droopFit.Fit(30, samples); // should take way less than 30 iterations to converge
//
//	meanDr = droopFit.params.dr;
//	// convert into incorporation-space from flow space
//	if (droopFit.params.base > 0.0)
//		meanDr /= droopFit.params.base; // assumption here is that the base value represents the avg signal initially in the region, so should be close to the 9/16 theoretical, but this calculation allows a better fit over all genomes
//	// printf("Fit droop: %.5lf base: %.5lf\n", meanDr, droopFit.params.base);
//
//	return meanDr;
//}

double CafieSolver::EstimateLibraryDroop(double *measuredVals, int numMeasured, double *stderr, int regionID)
{
	bool diagDump = false;
	int start = 16; // start after the key
	int end = std::min(116, numMeasured);
	int numSamples = end - start;
	std::vector<float> samples;
	samples.reserve(numSamples);

	if (diagDump) {
	    // dump out the signals used for the fit
	    std::ofstream signal_file("droop_estimate_signals.txt", std::ios_base::app);
	    signal_file << regionID << ", ";
	    for(int i=start;i<numMeasured;i++)
		    signal_file << fixed << measuredVals[i] << " ";
	    signal_file << std::endl;
	    signal_file.close();
    }

	for(int i=0;i<numSamples;i++) {
		if (measuredVals[i+start] < 1.0)
			samples.push_back(measuredVals[i+start]);
	}

	if (samples.size() < 4)
	{
		meanDr = 0.0015;
		if (stderr)
			*stderr = 0.0015;
		return meanDr;  // not much we can do to estimate with so little data
	}

	DroopFit droopFit(samples.size());
	DroopParams drMin, drMax;
	drMin.dr = 0.0;
	drMin.base = 0.4;
	drMax.dr = 0.01;
	drMax.base = 0.8;
	droopFit.params.dr = 0.0; // initial guess
	float baseline = (samples[0] + samples[1] + samples[2] + samples[3]) / 4.0;
	droopFit.params.base = (baseline > drMax.base ? drMax.base : baseline);

	droopFit.SetParamMin(drMin);
	droopFit.SetParamMax(drMax);

	droopFit.Fit(1000, &samples[0]); // should take way less than 1000 iterations to converge

	if (diagDump) {
	    // dump out the fit parameters
	    std::ofstream param_file("droop_estimate_params.txt", std::ios_base::app);
	    param_file << regionID << ", ";
	    param_file << numSamples << " ";
	    param_file << fixed << droopFit.params.base << " " << fixed << droopFit.paramsStdErr.base << " ";
	    param_file << fixed << droopFit.params.dr << " " << fixed << droopFit.paramsStdErr.dr << " ";
	    param_file << fixed << meanDr << std::endl;
	    param_file.close();
    }

	meanDr = droopFit.params.dr;
	if (droopFit.params.base > 0.0)
		meanDr = droopFit.params.dr / droopFit.params.base;

	if (stderr)
	{
		*stderr = droopFit.paramsStdErr.dr;
		if (droopFit.params.base > 0.0)
			*stderr = droopFit.paramsStdErr.dr / droopFit.params.base;
	}
	return meanDr;
}



// Marcin - temporary copy of EstimateLibraryDroop including modifications and fixes
double CafieSolver::EstimateLibraryDroop2(double *measuredVals, int numMeasured, double *stderr, int regionID)
{
	bool diagDump = false;
	int start = 16; // start after the key
	int end = std::min(116, numMeasured);
	int numSamples = end - start;
	std::vector<float> samples;
	std::vector<int> sampleTimePoint;
	samples.reserve(numSamples);
	sampleTimePoint.reserve(numSamples);

	if (diagDump) {
	    // dump out the signals used for the fit
	    std::ofstream signal_file("droop_estimate_signals.txt", std::ios_base::app);
	    signal_file << regionID << ", ";
	    for(int i=start;i<numMeasured;i++)
		    signal_file << fixed << measuredVals[i] << " ";
	    signal_file << std::endl;
	    signal_file.close();
    }

	for(int i=0;i<numSamples;i++) {
		if (measuredVals[i+start] < 2.0) {
			samples.push_back(measuredVals[i+start]);
			sampleTimePoint.push_back(i+1);
		}
	}

	if (samples.size() < 4)
	{
		meanDr = 0.0015;
		if (stderr)
			*stderr = 0.0015;
		return meanDr;  // not much we can do to estimate with so little data
	}

	DroopFit droopFit(samples.size());
	DroopParams drMin, drMax;
	drMin.dr = 0.0;
	drMin.base = 0.4;
	drMax.dr = 0.01;
	drMax.base = 0.8;
	droopFit.params.dr = 0.0; // initial guess
	float baseline = (samples[0] + samples[1] + samples[2] + samples[3]) / 4.0;
	droopFit.params.base = (baseline > drMax.base ? drMax.base : baseline);

	droopFit.SetParamMin(drMin);
	droopFit.SetParamMax(drMax);

	droopFit.SetTimePoint(sampleTimePoint);

	droopFit.Fit(1000, &samples[0]); // should take way less than 1000 iterations to converge

	if (diagDump) {
	    // dump out the fit parameters
	    std::ofstream param_file("droop_estimate_params.txt", std::ios_base::app);
	    param_file << regionID << ", ";
	    param_file << numSamples << " ";
	    param_file << fixed << droopFit.params.base << " " << fixed << droopFit.paramsStdErr.base << " ";
	    param_file << fixed << droopFit.params.dr << " " << fixed << droopFit.paramsStdErr.dr << " ";
	    param_file << fixed << meanDr << std::endl;
	    param_file.close();
    }

	meanDr = droopFit.params.dr;
	if (droopFit.params.base > 0.0)
		meanDr = droopFit.params.dr / droopFit.params.base;

	if (stderr)
	{
		*stderr = droopFit.paramsStdErr.dr;
		if (droopFit.params.base > 0.0)
			*stderr = droopFit.paramsStdErr.dr / droopFit.params.base;
	}
	return meanDr;
}




double CafieSolver::EstimateDroop(double *measuredVals, int numMeasured, int *expected)
{
	// simple sliding window approach
	// window has an initial span, and slides along all measured flows while also increasing its span size
	double windowWidth = 9.0;
	double windowInc = 0.1;

	bool done = false;
	int start = 0, end, width;
	double avg;
	// double avgDroop = 0.0;
	// int avgDroopCount = 0;
	double avgList[numMeasured];
	int avgListNum = 0;
	int i;
	double m;
	int count;
	while (!done) {
		width = (int)windowWidth;
		end = start + width;
		if (end > numMeasured) {
			end = numMeasured;
			width = end - start + 1;
		}
		avg = 0.0;
		count = 0;
		for(i=start;i<end;i++) {
			m = measuredVals[i];
			bool use = false;
			if (expected) {
				if (expected[i] == 1)
					use = true;
			} else {
				if (m >= 0.3 && m <= 1.3)
					use = true;
			}

			if (use) {
				avg += m;
				count++;
			}
		}
		if (count > 0) {
			avg /= (double)count;
			avgList[avgListNum] = avg;
			avgListNum++;
		}

		start++;
		windowWidth += windowInc;

		if (end >= numMeasured)
			done = true;
	}

	// so now we have an array of signal measures over time, lets do a linear estimate of the droop
	// first approach - just take the avg of the last 3 values divided by the avg of the first 3, thats our droop
	double droopEst = 0.0;
	if (avgListNum > 4) {
		double sig1 = (avgList[0] + avgList[1] + avgList[2]);
		double sig2 = (avgList[avgListNum-1] + avgList[avgListNum-2] + avgList[avgListNum-3]);
		droopEst = (sig1 == 0.0 ? 0.0 : 1.0 - (sig2/sig1));
	}

	return droopEst;
}

void CafieSolver::SimulateCAFIE(double simcaf, double simie, double simdr, int numFlows)
{
	// initialize the simulation states
	InitExtensionSim(&extSim);
	int i;
	for(i=0;i<numFlows;i++) {
		predicted[i] = ApplyReagent(&extSim, dnaTemplate, testLen, i, simie, simcaf, simdr, false);
		CompactStates(&extSim);
	}
}

void CafieSolver::GetCall(int callNum, DNATemplate *call)
{
	if (callNum >= 0 && callNum < numCalls) {
		call->base = dnaTemplate[callNum].base;
		call->count = dnaTemplate[callNum].count;
	}
}

/*const char *CafieSolver::GetSequence(DNATemplate *dna, int dnalen)
{
	// generate, then return a pointer to the sequence string

	if (dna == NULL) {
		dna = dnaTemplate;
		dnalen = testLen;
	}

	if (seqString)
		delete [] seqString;

	int len = 0;
	int i;
	for(i=0;i<dnalen;i++)
		len += dna[i].count;
	seqString = new char[len+1];

	len = 0;
	for(i=0;i<dnalen;i++) {
		int j;
		for(j=0;j<dna[i].count;j++) {
			seqString[len] = dna[i].base;
			len++;
		}
	}
	seqString[len] = NULL; // make it play nice as a 'C' string

	return seqString;
}*/

void CafieSolver::SetMeasured(int numMeasured, double *measuredVals)
{
	numFlows = numMeasured;
	memcpy(measured, measuredVals, numFlows * sizeof(double));
	memcpy(origMeasured, measuredVals, numFlows * sizeof(double));
	numCalls = 0;

	// validate the input
	int i;
	bool ok = true;
	for(i=0;i<numFlows;i++) {
		// check for nan and inf values in the measured values array
		if (std::isnan(measured[i]) || std::isinf(measured[i])) {
			measured[i] = 0.0; // just set to 0 in case other code tries to use this anyway.  We will set numFlows to 0 below
			ok = false;
		}
	}
	if (!ok)
		numFlows = 0;
}

void CafieSolver::SetMeasured(int numMeasured, float *measuredValsFloat)
{
	double *measuredValsDouble = new double[numMeasured];
	for(int i=0; i<numMeasured; i++)
		measuredValsDouble[i] = (double) measuredValsFloat[i];
	SetMeasured(numMeasured, measuredValsDouble);
	delete [] measuredValsDouble;
}

void CafieSolver::Normalize(char *keySequence, double droopEst)
{
//	int i;

        int flows = 0;
        int bases = 0;
	int numKeyFlows = 7;
	int ionogram[numKeyFlows];
	int len = strlen(keySequence);
        while (flows < numKeyFlows && bases < len) {
                ionogram[flows] = 0;
                while (flowOrder[flows%numFlowsPerCycle] == keySequence[bases] && bases < len) {
                        ionogram[flows]++;
                        bases++;
                }
                flows++;
        }

	Normalize(ionogram, numKeyFlows, droopEst, false);
}

void CafieSolver::PerNucShift(double s[4])
{
	for(int flow=0; flow<numFlows; ++flow)
		measured[flow] += s[flow%4];
}

double CafieSolver::Normalize(int *keyVec, int numKeyFlows, double droopEst, bool removeZeroNoise, bool perNuc)
{
	int i;
	int oMers[4][100];
	int zMers[4][100];
//	double renorm[16];
	int oCnt[4];
	int zCnt[4];

	memset(oMers, 0, sizeof(oMers));
	memset(zMers, 0, sizeof(zMers));
	memset(oCnt, 0, sizeof(oCnt));
	memset(zCnt, 0, sizeof(zCnt));

	int nuc;

	for (i=0;i<numKeyFlows;i++)
	{
		if (perNuc)
			nuc = GetNuc(i);
		else
			nuc = 0;
		if (keyVec[i] == 1) {
			oMers[nuc][oCnt[nuc]] = i;
			oCnt[nuc]++;
		} else if (keyVec[i] == 0) {
			zMers[nuc][zCnt[nuc]] = i;
			zCnt[nuc]++;
		}
	}

	// optionally remove 0-mer 'noise'
	if (removeZeroNoise) {
		for(nuc=0;nuc<(perNuc?4:1);nuc++) {
			double zeroMer = 0.0;
			double zeroMerMult = 0.0;
			if (zCnt[nuc] > 0)
				zeroMerMult = 1.0/zCnt[nuc];
			for(i=0;i<zCnt[nuc];i++)
				zeroMer += zeroMerMult * measured[zMers[nuc][i]];

			for(i=nuc;i<numFlows;i += (perNuc?4:1))
				measured[i] -= zeroMer;
		}
	}

	// correct for droop
	droopEst = 1.0 - droopEst;
	double droop = 1.0;
	for(i=0;i<numFlows;i++) {
		measured[i] *= droop;
		droop /= droopEst;
	}

	// first pass normalize to avg 1-mer
	double oneMer = 0.0;
	for(nuc=0;nuc<(perNuc?4:1);nuc++) {
		oneMer = 0.0;
		double oneMerMult = 0.0;
		if (oCnt[nuc] > 0)
			oneMerMult = 1.0 / oCnt[nuc];

		for(i=0;i<oCnt[nuc];i++)
			oneMer += oneMerMult * measured[oMers[nuc][i]];

		if (oneMer > 0.0) {
			for(i=nuc;i<numFlows;i += (perNuc?4:1)) {
				if (!std::isnan(measured[i] / oneMer)) {
					measured[i] /= oneMer;
				} else {
				//fprintf (stderr, "%s generated a NaN\n", "CafieSolver::Normalize");
				}
			}
			multiplier /= oneMer;
		}
	}
/*
	// normalize across first 12 flows, making best-guess calls as we go
	// do it in a couple iterations to make it a little more accurate
	double sf = 0.0;
	for (int iter=0;iter < 3;iter++)
	{
		// do the re-normalization with the current scale factor
		for (i=0;i < 12;i++)
		{
			if (measured[i] >= 1.0)
				renorm[i] = measured[i] - 2.0*sf;
			else
				renorm[i] = measured[i] - 2.0*sf*measured[i];
		}

		// calculate new scale factor
		sf = 0.0;
		double sfcnt = 0.0;
		for (i=0;i < 12;i++)
		{
			if (measured[i] > 0.5)
			{
				double err=measured[i]-(double)((int)(measured[i]+0.5));
				sf += err*measured[i];
				sfcnt += measured[i];
			}
		}

		sf /= sfcnt;
	}

	// Apply the correction
	for (i=0;i < numFlows;i++)
	{
		if (measured[i] >= 1.0)
			measured[i] = measured[i] - 2.0*sf;
		else
			measured[i] = measured[i] - 2.0*sf*measured[i];
	}
*/

	double sf = 0.0;

	return (oneMer-2.0*sf); // MGD - not sure who was looking at this return value, but its not going to work with the per-nuc option now
}


void CafieSolver::PerNucScale(double s[4])
{
	int i;
	for(i=0;i<numFlows;i++)
		measured[i] *= s[i%4];
}

int CafieSolver::FixDots(DNATemplate *dnaTemplateGuess, double predictedflowValue[MAX_TEMPLATE_SIZE][MAX_MER], int calls)
{
	int i;
	int curCall = 0;
	for(i=0;i<numFlows;i++) {
		if ((i > 2) && predictedExtension[i-2] == 0 && predictedExtension[i-1] == 0 && predictedExtension[i] == 0) {
			int maxFlowIndex = predictedflowValue[i-2][0] > predictedflowValue[i-1][0] ? i-2 : i-1;
			maxFlowIndex = predictedflowValue[maxFlowIndex][0] > predictedflowValue[i][0] ? maxFlowIndex : i;

			predictedExtension[maxFlowIndex] = 1;

			// move out all bases from here to end
			int m;
			for(m=calls;m>curCall;m--) {
				dnaTemplateGuess[m].base = dnaTemplateGuess[m-1].base;
				dnaTemplateGuess[m].count = dnaTemplateGuess[m-1].count;
			}
			calls++;
			// set new inserted base
			dnaTemplateGuess[curCall].base = flowOrder[i%numFlowsPerCycle];
			dnaTemplateGuess[curCall].count = 1;
			curCall++;
		} else {
			if (predictedExtension[i])
				curCall++;
		}
	}

	return curCall;
}


void CafieSolver::ResidualScale() {
	bool nucSpecific=false;
	int minFlow = 12;
	int maxFlow = 50;
	int nWeight=MAX_MER;
	double weight[MAX_MER];
	for(int i=0; i<MAX_MER; i++)
		weight[i] = 0;
	weight[1] = 1;
	weight[2] = 0.8;
	weight[3] = 0.64;
	weight[4] = 0.512;
	double minObsValue=MIN_NORMALIZED_SIGNAL_VALUE;

	ResidualScale(nucSpecific, minFlow, maxFlow, weight, nWeight, minObsValue);
}

void CafieSolver::ResidualScale(bool nucSpecific, int minFlow, int maxFlow, double *weight, int nWeight, double minObsValue)
{
	// Rescale intensities 
	if(maxFlow >= numFlows)
		maxFlow = numFlows-1;
	if(minFlow > maxFlow)
		minFlow = maxFlow;
	if(!nucSpecific) {
		double thisMult=1;
		double residual_numerator=0;
		double residual_denominator=0;
		for(int i=minFlow; i<=maxFlow; i++) {
			if(predictedExtension[i] < nWeight) {
				if(std::isnan(measured[i]) || std::isnan(predicted[i]))
					continue;
				double meas = std::max(measured[i],minObsValue);
				double pred = std::max(predicted[i],minObsValue);
				double logRatio = log2(meas/pred);
				double thisWeight = weight[predictedExtension[i]];
				residual_numerator += logRatio * thisWeight;
				residual_denominator += thisWeight;
			}
		}
		if(residual_denominator > 0) {
			thisMult = pow(2.0,-1*residual_numerator/residual_denominator);
			for(int i=0; i<numFlows; i++)
				measured[i] = thisMult * measured[i];
			multiplier *= thisMult;
		}
	}
}

double CafieSolver::Solve(int recallBases, bool fixDots)
{
	double hpSignal[MAX_MER];
	double sigMult=1.0;
	double doScale=false;

	for(int i=0; i < MAX_MER; i++)
		hpSignal[i] = i;
	
	return(Solve(recallBases,hpSignal,MAX_MER,sigMult,doScale,fixDots));
}

// Solve - corrects the normalized measured values for CAF & IE values
double CafieSolver::Solve(int recallBases, double *hpSignal, int nHpSignal, double sigMult, bool doScale, bool fixDots)
{
	int dotFixCorrectionCount = 0;
	int recallCorrectionCount = 0;

	// initialize the simulation states
	InitExtensionSim(&extSim);

	// -------------------------------------------------------------------------------
	// first pass calls all bases, essentially using just incomplete extension effects
	// -------------------------------------------------------------------------------

	// initialize our DNA template guess array
	DNATemplate dnaTemplateGuess[MAX_TEMPLATE_SIZE];
	memset(dnaTemplateGuess, 0, sizeof(dnaTemplateGuess));

	// start with no calls made, so assume nothing about the measured sequence before us
	numCalls = 0;

	// we keep track of the predicted flow values for each flow for each n-mer we predict
	double predictedflowValue[MAX_TEMPLATE_SIZE][MAX_MER];
	memset(predictedflowValue, 0, sizeof(predictedflowValue));
	memset(predictedExtension, 0, sizeof(predictedExtension));

	int i;
	int ext;
	int predictedExt;
	double error, minError;
	double predictedDelta[MAX_TEMPLATE_SIZE]; // an interesting feature is that for the 0-mer, we predict some base value, then for the 1-mer, 2-mer, etc we predict the base + N-mer * delta
	// note on above - right now we sill run through all N-mers!  performance increase by just doing the 0-mer and 1-mer and calculating the delta

	for(i=0;i<numFlows;i++) {

		// establish the signal for the 0-mer case
		dnaTemplateGuess[numCalls].base = 'N';
		dnaTemplateGuess[numCalls].count = 0;
#ifdef ACCURATE
		predictedflowValue[i][0] = ApplyReagent(&extSim, dnaTemplateGuess, numCalls+1, i, meanIe, meanCaf, meanDr, true, hpSignal, nHpSignal, sigMult);
#else
		predictedflowValue[i][0] = ApplyReagentFast(&extSim, dnaTemplateGuess, numCalls+1, i, meanIe, meanCaf, meanDr, (i>0), true, hpSignal, nHpSignal, sigMult);
#endif /* ACCURATE */
		predictedExt = 0;
		error = measured[i] - predictedflowValue[i][0];
		minError = error * error;

		// now look at the 1-mer and on up, and bail out when we go past our best (minimal error) solution
		for(ext=1;ext<MAX_MER;ext++) {
			// make a prediction for each n-mer, one at a time, and store our predicted signal, to later compare with the measured signal at that flow
			dnaTemplateGuess[numCalls].base = flowOrder[i%numFlowsPerCycle]; // this causes a match against the next reagent flow
			dnaTemplateGuess[numCalls].count = ext;

			// get a signal prediction based on the current model (extSim) and our current dna template guess
#ifdef ACCURATE
			predictedflowValue[i][ext] = ApplyReagent(&extSim, dnaTemplateGuess, numCalls+1, i, meanIe, meanCaf, meanDr, true, hpSignal, nHpSignal, sigMult);
#else
			predictedflowValue[i][ext] = ApplyReagentFast(&extSim, dnaTemplateGuess, numCalls+1, i, meanIe, meanCaf, meanDr, (i>0), true, hpSignal, nHpSignal, sigMult);
#endif /* ACCURATE */

			// check the error on this prediction, if worse than previous error then we've gone too far (up the n-mer trail) so previous prediction is the one we want
			error = measured[i] - predictedflowValue[i][ext];
			error = error * error;
			if (error < minError) {
				minError = error;
				predictedExt = ext;
			} else { // error now increasing again, so bail
				break;
			}
		}

		// not crazy about this hard-coded delta, it won't work correctly with non-linear hp scaling for example
		// also concerned that its not reflective of whats really going on with CF, might need the alternate approach below
		// predictedDelta[i] = predictedflowValue[i][1] - predictedflowValue[i][0];
		// alternate method would be this:
		int pe = predictedExt > 0 ? predictedExt : 1; // for predicted extensions of 0-mer, we still want 1-mer minus 0-mer
		predictedDelta[i] = predictedflowValue[i][pe] - predictedflowValue[i][pe-1];
		if (predictedDelta[i] == 0.0)
			predictedDelta[i] = 1.0;
		// assert(predictedDelta[i] > 0.0);

		// if we predict a 1-mer or more, add that call to our template guesss
		if (predictedExt > 0) {
			dnaTemplateGuess[numCalls].base = flowOrder[i%numFlowsPerCycle];
			dnaTemplateGuess[numCalls].count = predictedExt;
			dnaTemplateGuess[numCalls].qual = (predictedflowValue[i][predictedExt] - measured[i]) / predictedDelta[i];
			numCalls++;
		}

		// apply the reagent to update our model
#ifdef ACCURATE
		ApplyReagent(&extSim, dnaTemplateGuess, numCalls, i, meanIe, meanCaf, meanDr, false, hpSignal, nHpSignal, sigMult);
#else
		ApplyReagentFast(&extSim, dnaTemplateGuess, numCalls, i, meanIe, meanCaf, meanDr, (i>0), false, hpSignal, nHpSignal, sigMult);
#endif /* ACCURATE */
		CompactStates(&extSim);

		// store the predicted extension and signal at this flow
		predictedExtension[i] = predictedExt;
		predicted[i] = predictedflowValue[i][predictedExt];
	}
	// printf("After 1 iteration, sequence:\n<%s>\n", GetSequence(dnaTemplateGuess, numCalls));
	// printf("<%s>\n", GetSequence(dnaTemplate, testLen));

	// -------------------------------------------------------------------------------
	// re-calling to correct for carry-forward
	// here, we have a pretty good guess of our sequence, this refines it by using the
	// 'future' calls to help correct for carry-forward
	// -------------------------------------------------------------------------------

	ExtensionSim bestEstimateSim2;
	bestEstimateSim2.numStates = 0;

// printf("num calls: %d\n", numCalls);

	// numCalls = FixDots(dnaTemplateGuess, predictedflowValue, numCalls);
// printf("num calls: %d\n", numCalls);
// printf("After fixed 1 iteration, sequence:\n<%s>\n", GetSequence(dnaTemplateGuess, numCalls));

	int lastDotFixedPos;
	int fixedRunningCount; // counts sequential dot fixes, if this exceeds 10 we are probably done with the read so bail

	bool anyChanges = true;
	while (recallBases > 0 && anyChanges) {
		anyChanges = false; // assume we won't do anything on this pass and we can then bail early
		dotDetectionFlows.clear();
		dotPromotionFlows.clear();
		lastDotFixedPos = -1;
		fixedRunningCount = 0;
		dotFixCorrectionCount = 0;

		if(doScale) {
			ResidualScale();
		}
		InitExtensionSim(&bestEstimateSim2);
		int numCalls2, origBase, origExt;
		numCalls2 = 0;

		// loop through all flows, with a couple of early-out conditions:
		// 1: if fixedRunningCount exceeds our threshold, bail - this is an indicator that we have hit the end of the read as the counter tracks continuous dots
		// 2:  if dotFixCorrectionCount is more than 10% of the flows, its a low quality read or our corrections are not working, so bail since we are just f'in it up anyway
		for(i=0;(i<numFlows) && (fixedRunningCount < 5) && (dotFixCorrectionCount < numFlows/10);i++) {
			// save off original calls, since we will re-guess at this call, to see if we can change it
			origBase = dnaTemplateGuess[numCalls2].base;
			origExt = dnaTemplateGuess[numCalls2].count;

			dnaTemplateGuess[numCalls2].base = 'N';
			dnaTemplateGuess[numCalls2].count = 0;
#ifdef ACCURATE
			predictedflowValue[i][0] = ApplyReagent(&bestEstimateSim2, dnaTemplateGuess, numCalls, i, meanIe, meanCaf, meanDr, true, hpSignal, nHpSignal, sigMult);
#else
			predictedflowValue[i][0] = ApplyReagentFast(&bestEstimateSim2, dnaTemplateGuess, numCalls, i, meanIe, meanCaf, meanDr, (i>0), true, hpSignal, nHpSignal, sigMult);
#endif /* ACCURATE */
			predictedExt = 0;
			error = measured[i] - predictedflowValue[i][0];
			minError = error * error;

			for(ext=1;ext<MAX_MER;ext++) {
				// set up to predict each n-mer
				dnaTemplateGuess[numCalls2].base = flowOrder[i%numFlowsPerCycle]; // this causes a match against the next reagent flow
				dnaTemplateGuess[numCalls2].count = ext;

				// predict, but now we can use the entire guessed sequence, so much better accounting for carry-forward
#ifdef ACCURATE
				predictedflowValue[i][ext] = ApplyReagent(&bestEstimateSim2, dnaTemplateGuess, numCalls, i, meanIe, meanCaf, meanDr, true, hpSignal, nHpSignal, sigMult);
#else
				predictedflowValue[i][ext] = ApplyReagentFast(&bestEstimateSim2, dnaTemplateGuess, numCalls, i, meanIe, meanCaf, meanDr, (i>0), true, hpSignal, nHpSignal, sigMult);
#endif /* ACCURATE */
// printf("%.3lf\t", predictedflowValue[i][ext]);

				// check the error on this prediction, if worse than previous error then we've gone too far (up the n-mer trail) so previous prediction is the one we want
				error = measured[i] - predictedflowValue[i][ext];
				error = error * error;
				if (error < minError) {
					minError = error;
					predictedExt = ext;
				} else { // error now increasing again, so bail
					break;
				}
			}

			// not crazy about this hard-coded delta, it won't work correctly with non-linear hp scaling for example
			// also concerned that its not reflective of whats really going on with CF, might need the alternate approach below
			// predictedDelta[i] = predictedflowValue[i][1] - predictedflowValue[i][0];
			// alternate method would be this:
			int pe = predictedExt > 0 ? predictedExt : 1; // for predicted extensions of 0-mer, we still want 1-mer minus 0-mer
			predictedDelta[i] = predictedflowValue[i][pe] - predictedflowValue[i][pe-1];

			// assert(predictedDelta[i] > 0.0);
			if (predictedDelta[i] == 0.0)
				predictedDelta[i] = 1.0;
//printf("\n");
//if (predictedflowValue[i][0] == predictedflowValue[i][1])
//printf("Bad\n");


			// here we will compare our new predicted call against that of previous passes
			// we will then have no change, or an insert, or a delete, and will update the
			// current best guess (dnaTemplateGuess) accordingly

			if (predictedExt > 0) {
				if (predictedExtension[i] == 0) { // this is an insert!
					// move out all bases from here to end
					int m;
					for(m=numCalls;m>numCalls2;m--) {
						dnaTemplateGuess[m].base = dnaTemplateGuess[m-1].base;
						dnaTemplateGuess[m].count = dnaTemplateGuess[m-1].count;
						dnaTemplateGuess[m].qual = dnaTemplateGuess[m-1].qual;
					}
					numCalls++;
					// set new inserted base
					dnaTemplateGuess[numCalls2].base = flowOrder[i%numFlowsPerCycle];
					dnaTemplateGuess[numCalls2].count = predictedExt;
					dnaTemplateGuess[numCalls2].qual = (predictedflowValue[i][predictedExt] - measured[i])/predictedDelta[i];
					numCalls2++;
//printf("Insert.\n");
				} else { // this is an adjustment or no change
					dnaTemplateGuess[numCalls2].base = flowOrder[i%numFlowsPerCycle];
					dnaTemplateGuess[numCalls2].count = predictedExt;
					dnaTemplateGuess[numCalls2].qual = (predictedflowValue[i][predictedExt] - measured[i])/predictedDelta[i]; // could just use error here
					numCalls2++;
				}
			} else {
				if (predictedExtension[i] > 0) { // this is a deletion!
					// move all bases from here to end back one (essentially overwrite the base at numCalls2)
					int m;
					for(m=numCalls2+1;m<numCalls;m++) {
						dnaTemplateGuess[m-1].base = dnaTemplateGuess[m].base;
						dnaTemplateGuess[m-1].count = dnaTemplateGuess[m].count;
						dnaTemplateGuess[m-1].qual = dnaTemplateGuess[m].qual;
					}
					numCalls--;
//printf("Delete.\n");
				} else { // no change
					dnaTemplateGuess[numCalls2].base = origBase;
					dnaTemplateGuess[numCalls2].count = origExt;
				}
			}

			bool fixed = false;
			// fix dots
			int flowsBack = 0;
			if (fixDots && (i>2) && predictedExt == 0)
				flowsBack = DotTest(i, predictedExtension);
				// returns 0 if no dot found else returns the number of flows to be examined for most likely 1-mer promote
				// for standard flow orders this will always return 2 when a 'dot' is detected
				// for flow orders such as TANGO, it can return greater than 2
			// if (fixDots && (i > 2) && predictedExtension[i-2] == 0 && predictedExtension[i-1] == 0 && predictedExt == 0) {
			if (flowsBack > 0) {
				dotDetectionFlows.push_back(i);
				predictedExtension[i] = predictedExt;
				// int maxFlowIndex = measured[i-2] > measured[i-1] ? i-2 : i-1;
				// maxFlowIndex = measured[maxFlowIndex] > measured[i] ? maxFlowIndex : i;
				int maxFlowIndex = i;
				int priorFlow;
				for(priorFlow=i-1;priorFlow>=(i-flowsBack);priorFlow--) {
					if (measured[priorFlow] > measured[maxFlowIndex]) {
					// MGD note - there may be other metrics rather than looking at the raw measured signals,
					// such as which 0-mer flow had the highest error (lowest confidence)
						maxFlowIndex = priorFlow;
					}
				}

				predictedExtension[maxFlowIndex] = 1;
				dotPromotionFlows.push_back(maxFlowIndex);
// printf("Dot found from pos %d, best fit is index %d\n", i, maxFlowIndex);
				if (i == maxFlowIndex)
					predictedExt = 1;

				// move out all bases from here to end
				int m;
				for(m=numCalls;m>numCalls2;m--) {
					dnaTemplateGuess[m].base = dnaTemplateGuess[m-1].base;
					dnaTemplateGuess[m].count = dnaTemplateGuess[m-1].count;
					dnaTemplateGuess[m].qual = dnaTemplateGuess[m-1].qual;
				}
				numCalls++;
				// set new inserted base
				dnaTemplateGuess[numCalls2].base = flowOrder[maxFlowIndex%numFlowsPerCycle];
				dnaTemplateGuess[numCalls2].count = 1;
				// OK, since quality was clearly 'low' on all N 0-mers, assign some low quality to this new 1-mer
				// alternate method may be to run as the inverse of the current 0-mer call quality
				dnaTemplateGuess[numCalls2].qual = 0.25;
				numCalls2++;
				fixed = true;
				dotFixCorrectionCount++;
				if (lastDotFixedPos > (i-(flowsBack+1))) { // if we called a 'dot' within the last 'flowsBack + 1' flows, normally 3
					fixedRunningCount++;
				} else
					fixedRunningCount = 1;
				lastDotFixedPos = i; // keep track of the most recent dot position
			}

			if (fixed || (predictedExtension[i] != predictedExt)) {
				bestEstimateSim2.numStates = 0;
				AddState(&bestEstimateSim2, strandCount, 0);
				int k;
				for(k=0;k<i;k++) {
#ifdef ACCURATE
					predictedflowValue[k][predictedExtension[k]] = ApplyReagent(&bestEstimateSim2, dnaTemplateGuess, numCalls, k, meanIe, meanCaf, meanDr, false, hpSignal, nHpSignal, sigMult);
#else
					predictedflowValue[k][predictedExtension[k]] = ApplyReagentFast(&bestEstimateSim2, dnaTemplateGuess, numCalls, k, meanIe, meanCaf, meanDr, (k>0), false, hpSignal, nHpSignal, sigMult);
#endif /* ACCURATE */
					if (k%2)
						CompactStates(&bestEstimateSim2);
				}
				if (k%2)
					CompactStates(&bestEstimateSim2);
				// CompactStates(&bestEstimateSim2);
// if (i > 40 && bestEstimateSim2.numStates == 1)
// printf("Here 1 at state %d\n", bestEstimateSim2.states[0].state);
				recallCorrectionCount++;

				anyChanges = true;
			}

			// apply the reagent to update our model
#ifdef ACCURATE
			predictedflowValue[i][predictedExt] = ApplyReagent(&bestEstimateSim2, dnaTemplateGuess, numCalls, i, meanIe, meanCaf, meanDr, false, hpSignal, nHpSignal, sigMult);
#else
//printf("State 1: %d\n", bestEstimateSim2.states[0].state);
			predictedflowValue[i][predictedExt] = ApplyReagentFast(&bestEstimateSim2, dnaTemplateGuess, numCalls, i, meanIe, meanCaf, meanDr, (i>0), false, hpSignal, nHpSignal, sigMult);
//printf("State 2: %d  val: %.3lf\n", bestEstimateSim2.states[0].state, predictedflowValue[i][predictedExt]);
// if (predictedflowValue[i][predictedExt] > 0.9)
//printf("Base: %c\n", flowOrder[i%4]);
#endif /* ACCURATE */
//printf("Sig: %.3lf\n", predictedflowValue[i][predictedExt]);
			CompactStates(&bestEstimateSim2);
// if (i > 40 && bestEstimateSim2.numStates == 1)
// printf("Here 2\n");

			// make sure we have the most recent prediction for this flow
			predictedExtension[i] = predictedExt;
		}
		recallBases--;

		// static int iterations = 1;
		// printf("After %d iterations, sequence, orig:\n<%s>\n", ++iterations, GetSequence(dnaTemplateGuess, numCalls2));
		// printf("<%s>\n", GetSequence(dnaTemplate, testLen));
// printf("numCalls: %d numCalls2: %d\n", numCalls, numCalls2);
		// if (fixDots)
			// numCalls = FixDots(dnaTemplateGuess, predictedflowValue, numCalls);
// printf("fixed num calls: %d\n", numCalls);
// printf("After fixed, sequence:\n<%s>\n", GetSequence(dnaTemplateGuess, numCalls));
	}

	// trim back to remove false 'dots' inserted
/*
	while ((numCalls > 0) && (dnaTemplateGuess[numCalls-1].qual > 0.5))
		numCalls--;
	if (numCalls < 0)
		numCalls = 0;
*/
	numCalls = 0;
	int numValidFlows = i; // the 'i' var will indicate the point where we possibly bailed early due to dot filter conditions met
	for(i=0;i<numValidFlows;i++) {
		if (predictedExtension[i] > 0) {
			numCalls++;
		}
	}

	// done, save off the template guess
	memcpy(dnaTemplate, dnaTemplateGuess, sizeof(DNATemplate) * numCalls);
	testLen = numCalls; // GetSequence uses testlen by default
	// printf("%s\n", GetSequence(dnaTemplate, testLen));

	// save off the corrected flowgram values, and calculate the error
	double deltaError, predictedError;
	predictedError = 0.0;
	int nextBase = 0;
	for(i=0;i<numFlows;i++) {
		if (predictedExtension[i] > 0) {
			deltaError = dnaTemplateGuess[nextBase].qual;
			nextBase++;
		} else
			deltaError = (predictedflowValue[i][predictedExtension[i]] - measured[i])/predictedDelta[i];
		if (deltaError < -0.49) deltaError = -0.49; // this case typically happens when we predict a 0-mer where the delta is also very small, so dividing by the delta causes the delta error to blow up and make things look like a corrected 1-mer or larger, but it should still be a 0-mer, just with a large error
		if (deltaError > 0.49) deltaError = 0.49;
		corrected[i] = predictedExtension[i] - deltaError;
		predicted[i] = predictedflowValue[i][predictedExtension[i]];
		predictedError += deltaError*deltaError;
	}
	if (numFlows > 0) {
		predictedError /= numFlows;
		predictedError = sqrt(predictedError);
	}
// printf("recalls: %d  dots: %d\n", recallCorrectionCount, dotFixCorrectionCount);
	return predictedError;
}

double CafieSolver::Solve2(int recallBases, bool fixDots)
{	//Unused parameter generates compiler warning, so...
	if (fixDots) {};
	
	double ideal[numFlows];
	double predictedDelta[numFlows];
	memset(ideal, 0, sizeof(double) * numFlows);

	ModelIdeal(ideal, 0, predicted, predictedDelta, numFlows, measured, numFlows, meanCaf, meanIe, meanDr);

	int i;
	for(i=0;i<recallBases;i++) {
		ModelIdeal(ideal, numFlows, predicted, predictedDelta, numFlows, measured, numFlows, meanCaf, meanIe, meanDr);
	}

	// calculate the corrected ionogram
	double scaledError;
	numCalls = 0;
	int flow;
	for(flow=0;flow<numFlows;flow++) {
		scaledError = (predicted[flow] - measured[flow])/predictedDelta[flow];
		corrected[flow] = ideal[flow] - scaledError;
		predictedExtension[flow] = (int)ideal[flow];
		numCalls += predictedExtension[flow];
	}

	return 0;
}

double	KeySNR(double *measured, int *keyVec, int numKeyFlows) {
  double zeroMerSig,zeroMerSD,oneMerSig,oneMerSD,keySig,keySD;
  return(KeySNR(measured, keyVec, numKeyFlows, &zeroMerSig, &zeroMerSD, &oneMerSig, &oneMerSD, &keySig, &keySD));
}

double	KeySNR(vector<weight_t> &measured_vec, int *keyVec, int numKeyFlows, double *zeroMerSig, double *zeroMerSD, double *oneMerSig, double *oneMerSD, double *keySig, double *keySD, double minSD) {
  double *measured = new double[measured_vec.size()];
  for(unsigned int iFlow=0; iFlow < measured_vec.size(); iFlow++)
    measured[iFlow] = measured_vec[iFlow];
  double val = KeySNR(measured, keyVec, numKeyFlows, zeroMerSig, zeroMerSD, oneMerSig, oneMerSD, keySig, keySD, minSD);
  delete [] measured;
  return(val);
}

double	KeySNR(double *measured, int *keyVec, int numKeyFlows, double *zeroMerSig, double *zeroMerSD, double *oneMerSig, double *oneMerSD, double *keySig, double *keySD, double minSD) {
  double *zeroMer= new double[numKeyFlows];
  int nZeroMer = 0;
  double *oneMer= new double[numKeyFlows];
  int nOneMer = 0;

  for(int i=0; i<numKeyFlows; i++) {
    if(keyVec[i] == 0) {
      zeroMer[nZeroMer++] = measured[i];
    } else if(keyVec[i] == 1) {
      oneMer[nOneMer++] = measured[i];
    }
  }

  *zeroMerSig = ionStats::median(zeroMer,nZeroMer);
  *zeroMerSD  = std::max(minSD,ionStats::sd(zeroMer,nZeroMer));
  *oneMerSig  = ionStats::median(oneMer,nOneMer);
  *oneMerSD   = std::max(minSD,ionStats::sd(oneMer,nOneMer));
  *keySig     = *oneMerSig - *zeroMerSig;
  *keySD      = sqrt(pow(*zeroMerSD,2) + pow(*oneMerSD,2));
  return(*keySig / *keySD);
}

float GetResidualSummary(float *measured, float *predicted, int minFlow, int maxFlow) {

  assert(maxFlow > minFlow);
  int nFlow = maxFlow-minFlow;
  float *residual = new float[nFlow];

  for(int iRes=0,iFlow=minFlow; iRes<nFlow; iRes++,iFlow++) {
    float m = measured[iFlow];
    float p = predicted[iFlow];
    float r = fabs(m-p);
    residual[iRes] = r;
    //residual[iRes] = fabs(measured[iFlow] - predicted[iFlow]);
  }
  float median_residual = ionStats::median(residual,nFlow);

  delete [] residual;

  return(median_residual);
}

void SetFlowOrderIndex(int *flowOrderIndex, char *flowOrder, int numFlowsPerCycle) {
        int i;
        for(i=0;i<numFlowsPerCycle;i++) {
                if (flowOrder[i] == 'T')
                        flowOrderIndex[i] = 0;
                if (flowOrder[i] == 'A')
                        flowOrderIndex[i] = 1;
                if (flowOrder[i] == 'C')
                        flowOrderIndex[i] = 2;
                if (flowOrder[i] == 'G')
                        flowOrderIndex[i] = 3;
        }
}
