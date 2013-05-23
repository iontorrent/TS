/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CAFIESOLVER_H
#define CAFIESOLVER_H

#include "RawWells.h"
#include <vector>
#include "ChipIdDecoder.h"

struct DNATemplate {
	int		base;
	int		count;
	double		qual;
};

struct ExtensionState {
	int	strands;
	int	state;
};


//Data structure for CafieSolver Threads
struct CafieThreadInfo {
	pthread_t id;
	pthread_mutex_t *job_queue_mutex;
	pthread_mutex_t *results_mutex;

	RawWells *rawWells;
	RawWells *residualWells;
	Mask *bfmask;
	MaskType readMask;
	char *flowOrder;
	int numFlows;
	int numCafieFlows;
	SequenceItem *seqList;
	int cafieYinc;
	int cafieXinc;
	int rows;
	double *nucMult;
	double *cf;
	double *ie;
	double *droop;
	double *flowMult;
	int KEYPASSFILTER;
	int *numReads;
	int *numClonal;
	int *numValidReads;
	int *numFailKeyBases;
	int *numFailPPF;
	int *numFailCafieRes;
	int *numZeroBases;
	int *numShortKeyBases;
	uint32_t clip_qual_left;
	uint32_t clip_qual_right;
	uint32_t clip_adapter_left;
	uint32_t clip_adapter_right;
	char *runId;
	bool NormalizeZeros;
	int SCALED_SOLVE2;
	FILE *wellStatFileFP;
	bool dotFixDebug;
	double hpScaleFactor;
	int wantPerWellCafie;
	int wantDotFixes;
	int NONLINEAR_HP_SCALE;
	int numFlowsPerCycle;
	int minReadLength;
	bool percentPositiveFlowFilterOn;
	std::vector<float> perFlowScaleVal;
	bool perFlowScale;
	int phredScoreVersion;
    std::string phredTableFile;
    
	double percentPositiveFlowsMaxValue;
	double percentPositiveFlowsMinValue;
	int cafieResFilterCalling;
	double cafieResMaxValue;
	bool clonalFilterSolve;
};

// list of dynamic quality filters used in the per-well SolveIdeal method
enum FilterTests {
	FilterTest_PerBaseQual = 1,
};

// Maximum length template we can handle (a 10-mer only counts as one entry also)
#define MAX_TEMPLATE_SIZE 1280
#define MAX_FLOWS 1600
#define MAX_MER 12

// A value for minimum key-normalized signal, used to avoid divide-by-small-number issues
#define MIN_NORMALIZED_SIGNAL_VALUE 0.01

// run with or without malloc
#define NO_MALLOC
#ifdef NO_MALLOC
// as reads get longer, number of states we need to keep track of grows, with no malloc, there needs to me a set limit
#define MAX_STATES 1000
#endif /* NO_MALLOC */

// allows our carry-forward portion to further possibly extend
#define CARRY_FORWARD_2

struct ExtensionSim {
#ifdef NO_MALLOC
	ExtensionState states[MAX_STATES];
#else
	ExtensionState *states;
#endif /* NO_MALLOC */
	int numStates;
};

struct IonogramGrid {
	char	seq[512];
	int	cfNum, ieNum;
	double	cfLow, cfHigh, cfInc;
	double	ieLow, ieHigh, ieInc;
	double  hpSignal[MAX_MER];
	double  sigMult;
	int	numFlows;
	double	**predictedValue; // 2D array of Ionograms - each Ionogram is of len: numFlows
};


class CafieSolver {
	public:
		CafieSolver();
		virtual ~CafieSolver();

		void	SetFlowOrder(char *flowOrder);
		void	SetMeasured(int numMeasured, double *measuredVals);
		void	SetMeasured(int numMeasured, float *measuredVals);
		const double	*GetMeasured() {return measured;}
		void	SetCAFIE(double caf, double ie) {meanCaf = caf; meanIe = ie;}
		void SetDroop(double dr) {meanDr = dr;}

		// Normalize - takes the measured values, and normalizes based on the key sequence
		void	Normalize(char *keySequence, double droopEst);
		double	Normalize(int *keyVec, int numKeyFlows, double droopEst, bool removeZeroNoise, bool perNuc = false);
		void	PerNucScale(double s[4]);
		void	PerNucShift(double s[4]);

		// Solve - corrects the normalized measured values for CAF & IE values
		// iterates to account for carry-forward effects, does this recallBases times
		double	Solve(int recallBases, double *hpSignal, int nHpSignal, double sigMult, bool doScale, bool fixDots = true);
		double	Solve(int recallBases, bool fixDots = true);
		double	Solve2(int recallBases, bool fixDots = true);
		void	ResidualScale();
		void	ResidualScale(bool nucSpecific, int minFlow, int maxFlow, double *weight, int nWeight, double minObsValue=MIN_NORMALIZED_SIGNAL_VALUE);

		int	SolveIdeal( double droopEstimate, int droopMode, bool reNormalize, ChipIdEnum phredTableVersion, bool scaledSolve = true, long unsigned int filterTests = FilterTest_PerBaseQual);
		void	ModelIdeal(double *ideal, int numIdealFlows, double *predicted, double *predictedDelta, int numPredictedFlows, double *measured, int numMeasured, double cfModeled, double ieModeled, double drModeled);

		// FindBestCAFIE - given a test sequence and a set of measured values, iteratively
		// solve for the best CAF & IE values that fit the measured values.
		double  FindBestCAFIE(double *measuredVals, double* predictedVals, int numVals);
		double	FindBestCAFIE(double *measuredVals, int numMeasured, bool useDroopEst, double droopEst, double *hpSignal, int nHpSignal, double sigMult);
		double	FindBestCAFIE(double *measuredVals, int numMeasured, bool useDroopEst, double droopEst);
		void	SetTestSequence(char *testSequence);
		void	SimulateCAFIE(double caf, double ie, double dr, int numFlows);
		void	SimulateCAFIE(double* predicted, const char* seq, const char* flowOrder, double cf, double ie, double dr, int nflows, double *hpSignal, int nHpSignal, double sigMult);
		void	SimulateCAFIE(double* predicted, const char* seq, const char* flowOrder, double cf, double ie, double dr, int nflows);

		// Model - you pass in the ideal Ionogram and num of Ionogram vals, and Model will return the vector of predicted values for the given cf, ie, dr params.  Note that there can be fewer predicted values than Ionogram values as you will get a falloff in prediction quality towards the end of the ionogram as carry-forward cannot be accounted for past the end of the true Ionogram.
		void	Model(double *ideal, int numIdealFlows, float *predicted, int numPredictedFlows, double cfModeled, double ieModeled, double drModeled);

		double	EstimateDroop(double *measuredVals, int numMeasured, int *expected);
		double	EstimateLibraryDroop(double *measuredVals, int numMeasured, double *stderr = NULL, int regionID=0);
		double	EstimateLibraryDroop2(double *measuredVals, int numMeasured, double *stderr = NULL, int regionID=0);

		// GetPredictedResult - returns what the solver expected to see as a measure for the incorporation length most closely matching the measured signal
		double	GetPredictedResult(int flowIndex) {return predicted[flowIndex];}
		double	*GetPredictedResult(void) {return predicted;}
		void    SetStrandCount(int count) {strandCount = count;}

		// GetCorrectedResult - Added the error (predicted - measured) to what the solver 'thinks' is the best called length, so good for plotting corrected Ionograms
		double	GetCorrectedResult(int flowIndex) {return corrected[flowIndex];}

		double	CAF() {return meanCaf;}
		double	IE() {return meanIe;}
		double	DR() {return meanDr;}
		int		GetNumCalls() {return numCalls;}
		void	GetCall(int callNum, DNATemplate *call);
		const char *GetSequence(DNATemplate *dnaTemplate = NULL, int dnalen = 0);
		int	GetPredictedExtension(int flowNum) {return predictedExtension[flowNum];}
		double	GetMultiplier() { return(multiplier); };
		std::vector<unsigned int> & GetDotDetectionFlows(void) { return(dotDetectionFlows); };
		std::vector<unsigned int> & GetDotPromotionFlows(void) { return(dotPromotionFlows); };

		// GetNuc - For a given flow & flow order, returns the nuc as an index 0 thru 3
		int	GetNuc(int flow);
		int	DotTest(int curFlow, int *predictedExtension);

	protected:
		void	AddState(ExtensionSim *extSim, int strands, int state);
		void inline CompactStates(ExtensionSim *extSim);
		double	ApplyReagent(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool testOnly, double *hpSignal, int nHpSignal, double sigMult);
		double	ApplyReagent(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int flowNum, double ie, double caf, double dr, bool testOnly);
		double	ApplyReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int r, double ie, double caf, double dr, bool applyCAF, bool testOnly, double *hpSignal, int nHpSignal, double sigMult);
		double	ApplyReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int r, double ie, double caf, double dr, bool applyCAF, bool testOnly = false);
		double	TestReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int r, double ie, double caf, double dr, bool applyCAF, double *hpSignal, int nHpSignal, double sigMult);
		double	TestReagentFast(ExtensionSim *extSim, DNATemplate *dnaTemplate, int numBases, int r, double ie, double caf, double dr, bool applyCAF);
		void	FreeExtensionSim(ExtensionSim *extSim);
		void	CopyExtensionSim(ExtensionSim *extSim, ExtensionSim *extSimCopy);
		void	InitExtensionSim(ExtensionSim *extSim);
		int	FixDots(DNATemplate *dnaTemplateGuess, double predictedflowValue[MAX_TEMPLATE_SIZE][MAX_MER], int calls);
		inline int DistToBase(char base, int flowNum);
		void	DistToBaseInit();

		int			numCalls;
		DNATemplate		dnaTemplate[MAX_TEMPLATE_SIZE];
		int			numTemplateBases;
		char			*flowOrder;
		int			numFlowsPerCycle;
		ExtensionSim		extSim;
		int			strandCount;
		int			testLen;
		double			multiplier; // used to accumulate whatever multiplication happends on the measured values
		double			meanCaf, meanIe, meanDr;
		double			predicted[MAX_TEMPLATE_SIZE];
		double			corrected[MAX_TEMPLATE_SIZE];
		int			numFlows;
		double			measured[MAX_FLOWS];
		double			origMeasured[MAX_FLOWS];
		char			*seqString;
		int			predictedExtension[MAX_TEMPLATE_SIZE];
		char			currentTestSequence[MAX_TEMPLATE_SIZE];
		int			*flowOrderIndex;
		static int		numIonogramGrids;
		static IonogramGrid	ionogramGrid[100];
		static int		instances;
		ExtensionSim		_localSim; // local method work var
		int			Dist[4][MAX_FLOWS];
		std::vector< unsigned int >  dotDetectionFlows;
		std::vector< unsigned int >  dotPromotionFlows;
};

double	KeySNR(double *measured, int *keyVec, int numKeyFlows);
double	KeySNR(std::vector<weight_t> &measured_vec, int *keyVec, int numKeyFlows, double *zeroMerSig, double *zeroMerSD, double *oneMerSig, double *oneMerSD, double *keySig, double *keySD, double minSD=0.01);
double	KeySNR(double *measured, int *keyVec, int numKeyFlows, double *zeroMerSig, double *zeroMerSD, double *oneMerSig, double *oneMerSD, double *keySig, double *keySD, double minSD=0.01);
float GetResidualSummary(float *measured, float *predicted, int minFlow, int maxFlow);
void SetFlowOrderIndex(int *flowOrderIndex, char *flowOrder, int numFlowsPerCycle);

#endif // CAFIESOLVER_H
