/* Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved */

#include "WellsNormalization.h"
// =================================================================================
ProcessingMask::ProcessingMask(const unsigned int numRows, const unsigned int numCols){
	SetMaskSize(numRows, numCols);
}

int ProcessingMask::SetMaskSize(const unsigned int numRows, const unsigned int numCols){
	numRows_ = numRows;
	numCols_ = numCols;
	try{
		//mask_.resize(numRows, vector<bool>(numCols, false));
		mask_.resize(numRows);
		for (size_t r=0; r < numRows; r++)
			mask_[r].resize(numCols, false);
	} catch (bad_alloc const&){
		return 1;
	}
	return 0;
}

bool ProcessingMask::Get(const unsigned int row, const unsigned int col) const{
	return mask_[row][col];
}

int ProcessingMask::Set(const unsigned int row, const unsigned int col, const bool value){
	mask_[row][col] = value;
	return 0; // success
}

unsigned int ProcessingMask::NumRows() const{
	return numRows_;
}

unsigned int ProcessingMask::NumCols() const{
	return numCols_;
}


//----------------------------------------------------------------------------------
Mark01Data::Mark01Data(const size_t rowStart, const size_t numRows, const size_t colStart, const size_t numCols, const size_t flowStart, const size_t numFlows, RawWells const *wells, ReadClassMap const *rcm){
	Initialize(rowStart, numRows, colStart, numCols, flowStart, numFlows, wells, rcm);
}

//----------------------------------------------------------------------------------
void Mark01Data::Initialize(const size_t rowStart, const size_t numRows, const size_t colStart, const size_t numCols, const size_t flowStart, const size_t numFlows, RawWells const *wells, ReadClassMap const *rcm){
	rowStart_ = rowStart;
	numRows_ = numRows;
	colStart_ = colStart;
	numCols_ = numCols;
	flowStart_ = flowStart;
	numFlows_ = numFlows;
	wells_ = wells;
	rcm_  =rcm;
	try {
	//m01_.resize( numRows , vector<vector<int> >( numCols,  vector<int> (numFlows, -1 )));
	//m01_res_.resize( numRows , vector<vector<float> >( numCols,  vector<float> (numFlows, 0. )));
		m01_.resize( numRows);
		m01_res_.resize( numRows);
		for (size_t r=0; r< numRows; r++){
			m01_[r].resize(numCols);
			m01_res_[r].resize(numCols, 1000.);
			for (size_t c = 0; c < numCols; c++){
				m01_[r][c].resize(numFlows, -1);
			}
		}
	} catch (bad_alloc const&){
		 ION_ABORT("Memory allocation failed Mark01Data in WellsNormalization!");
	}

}

// ---------------------------------------------------------------------------------
int& Mark01Data::category(const size_t row, const size_t col, const size_t flow){
	return m01_[row - rowStart_][col - colStart_][flow - flowStart_];
}

float& Mark01Data::residual(const size_t row, const size_t col){
	return m01_res_[row - rowStart_][col - colStart_];
}


float Mark01Data::MeanSignal(const int cat, const size_t flow, const ProcessingMask *mask){
	unsigned int f = flow - flowStart_;
	double signal = 0.;
	unsigned int count = 0;
	for (unsigned int r=0; r<numRows_; ++r) {
		for (unsigned int c=0; c<numCols_; ++c) {
			if (mask->Get(r,c) && m01_[r][c][f] == cat){
				signal += (double) wells_->At(rowStart_+r, colStart_+c, flowStart_ + f);
				count ++;
			}
		}
	}

	if (count >0){
		signal /= (double) count;
	} else {
		signal = 0.;
	}

	return (float) signal;
}

float Mark01Data::Fraction(const int cat, const size_t flow, const ProcessingMask *mask){
	unsigned int f = flow - flowStart_;
	unsigned int count = 0;
	unsigned int passedFilter = 0;
	for (unsigned int r=0; r<numRows_; ++r) {
		for (unsigned int c=0; c<numCols_; ++c) {
			if (mask->Get(r,c) && m01_[r][c][f] == cat){
				count ++;
			}
			if (m01_[r][c][f] != -1){
				passedFilter ++;
			}
		}
	}

	if (passedFilter > 0){
		return (float) ((double) count / (double) passedFilter);
	} else {
		return 0.f;
	}


}

int Mark01Data::Classify(const ProcessingMask *mask, const unsigned int numMem, const unsigned int goodFlowStart, const unsigned int goodFlowEnd){

	// Iterate over wells in this Chunk
	numWellsPassFilter = 0;
	double zeroSeed = 0.;
	double oneSeed = 1.;

	for (unsigned int r=0; r<numRows_; ++r) {
		for (unsigned int c=0; c<numCols_; ++c) {

//			if (colStart_+c== 100 && rowStart_ + r == 450){
//				cout<< "x:"<<colStart_+c <<" y:" << rowStart_+r << " mask " << mask[r][c] << endl;;
//			}

			// Discard well if it is not interesting of filtered
			if (! mask->Get(r,c)){
				for (size_t f= 0; f< numFlows_; f++){
					m01_[r][c][f] = -1;
				}
				continue;
			}

			numWellsPassFilter ++;

			unsigned int numZeros = 1;
			unsigned int numOnes = 1;
			double mem0[numMem], mem1[numMem];
			mem0[0] = zeroSeed;
			mem1[0] = oneSeed;

			double resAvg = 0.;  // average residual
			unsigned int resCount = 0;


			// classify 0 and 1 through
			for (size_t f= 0; f< numFlows_; f++){

				// calculate mean values of 0
				double mean0 = 0.;
				for (size_t i=0; i< min(numZeros, numMem); i++){
					mean0 += mem0[i];
				}
				mean0 /= (double) min(numZeros, numMem);

				// calculate mean values of 1
				double mean1 = 0.;
				for (size_t i=0; i< min(numOnes, numMem); i++){
					mean1 += mem1[i];
				}
				mean1 /= (double) min(numOnes, numMem);

				if (mean1 < mean0){
					m01_[r][c][f] = 0;
					continue;
				}


				// calculate distance from expected 0, 1 and 2-mer
				double flowValue = (double) wells_->At(rowStart_+r, colStart_+c, flowStart_ + f);
				double dist0 = fabs(flowValue - mean0);
				double dist1 = fabs(flowValue - mean1);
				double dist2 = fabs(flowValue - (2.*mean1-mean0));
				double res;


				// classify this flow
				if (dist0 < dist1){
					// 0-mer
					m01_[r][c][f] = 0;
					mem0[numZeros%numMem] = flowValue;
					numZeros ++;
					res = dist0;
				} else if (dist1 < dist2) {
					// 1-mer
					m01_[r][c][f] = 1;
					mem1[numOnes%numMem] = flowValue;
					numOnes ++;
					res = dist1;
				} else {
					// 2-mer or higher
					m01_[r][c][f] = 2;
					res = dist2;
				}

				// keep track of residuals of this well
				if (f>= goodFlowStart && f < goodFlowEnd){
					resAvg += res;
					resCount ++;
				}
			}

			if (resCount > 0)
				m01_res_[r][c] = resAvg/(double) resCount;



		}
	}

	return numWellsPassFilter;

}


// =================================================================================

WellsNormalization::WellsNormalization()
    : norm_method_("off"), flow_order_(NULL)
{
    is_disabled_ = true;
    wells_       = NULL;
    rcm_         = NULL;
    m01data_     = NULL;
    pinZero_ = false;
    DEBUG_ = 0;
    maskAll_ = maskGood_ = NULL;
    wells_index_ = 0;

    doKeyNorm_        = true;
    doSignalBiasCorr_ = true;

    offsetT_ = offsetA_ = offsetC_ = offsetG_ = offsetAvg_ = 0.0;
    nucStrengthT_ = nucStrengthA_ = nucStrengthC_ = nucStrengthG_ = 0.0;
    avgZeroSubtracted_ = minZeroSubtracted_ = maxZeroSubtracted_ = 0.0;

    nucOffsetStartFlowUsed_ = nucOffsetEndFlowUsed_ = 0;
    zeroFlowStartUsed_ = zeroFlowStopUsed_ = numWellsSubtractedZero_ = 0;
    numPassFilter_ = numGood_ = 0;
};

WellsNormalization::WellsNormalization(ion::FlowOrder const  *flow_order,
                                       const string &norm_method)
    : norm_method_(norm_method), flow_order_(flow_order)
{
	is_disabled_ = true;
	wells_       = NULL;
	rcm_         = NULL;
	m01data_     = NULL;
	pinZero_ = false;
	DEBUG_ = 0;
	maskAll_ = maskGood_ = NULL;
	wells_index_ = 0;

	doKeyNorm_        = true;
	doSignalBiasCorr_ = true;

	offsetT_ = offsetA_ = offsetC_ = offsetG_ = offsetAvg_ = 0.0;
	nucStrengthT_ = nucStrengthA_ = nucStrengthC_ = nucStrengthG_ = 0.0;
	avgZeroSubtracted_ = minZeroSubtracted_ = maxZeroSubtracted_ = 0.0;

	nucOffsetStartFlowUsed_ = nucOffsetEndFlowUsed_ = 0;
	zeroFlowStartUsed_ = zeroFlowStopUsed_ = numWellsSubtractedZero_ = 0;
	numPassFilter_ = numGood_ = 0;
}

// ---------------------------------------------------------------------------------
WellsNormalization::~WellsNormalization(){
	delete m01data_;

}

// ---------------------------------------------------------------------------------

void WellsNormalization::SetFlowOrder(ion::FlowOrder const  *flow_order,
                                      const string &norm_method)
{
  flow_order_  = flow_order;
  norm_method_ = norm_method;
}

bool WellsNormalization::SetWells(RawWells *wells, ReadClassMap const *rcm, unsigned int wells_file_index)
{
	is_disabled_ = true;
	if (wells == NULL) {
		return false;
	}
	else if (rcm == NULL)
		return false;
	else {
		wells_ = wells;
		rcm_   = rcm;
		wells_index_ = wells_file_index;

		// default "on" option
		if (norm_method_ != "off"){
			is_disabled_ = false;
			doKeyNorm_ = true;
			doSignalBiasCorr_ = true;
			flowCorrectMethod_ = "fix01"; // default flow correction option
		}


		if (norm_method_ == "keyOnly"){
			doKeyNorm_ = true;
			doSignalBiasCorr_ = false;
		} else if (norm_method_ == "signalBiasOnly"){
			doKeyNorm_ = false;
			doSignalBiasCorr_ = true;
		} else if (norm_method_ == "pinZero"){
			flowCorrectMethod_ = "pinZero";
		} else if (norm_method_ == "fix01"){
			flowCorrectMethod_ = "fix01";
		} else if (norm_method_ == "offset"){
			flowCorrectMethod_ = "default";
		}

		return true;
	}
}

// ---------------------------------------------------------------------------------

bool  WellsNormalization::is_filtered(int x, int y) const
{
	//return (not rcm_->IsValidRead(x,y));
  return (rcm_->UseWells(x,y, wells_index_) == false);
}

// -----------------------------------------------------------------------------------
bool  WellsNormalization::is_filtered_libOnly(int x, int y) const
{
	return (is_filtered(x,y) or rcm_->ClassMatch(x,y, MapTF));
}


// ---------------------------------------------------------------------------------

void WellsNormalization::DoKeyNormalization(const vector<KeySequence> & keys)
{

	if (is_disabled_ or keys.size() != 2 or !doKeyNorm_)
		return;
	if (DEBUG_)
		printf("do KeyNormalization.\n");

	// Load information about the currently loaded well chunk
	WellChunk mChunk = wells_->GetChunk();
	int read_class = -1;
	float key_normalizer, signal_sum;

	vector<int> key_base_count(keys.size(), 0);
	for (unsigned int iKey=0; iKey< keys.size(); ++iKey){
		for (int flow=0; flow<keys[iKey].flows_length()-1; ++flow)
			key_base_count[iKey] += keys[iKey][flow];
	}

	// Iterate over wells in this Chunk
	for (unsigned int x=mChunk.colStart; x<mChunk.colStart+mChunk.colWidth; ++x) {
		for (unsigned int y=mChunk.rowStart; y<mChunk.rowStart+mChunk.rowHeight; ++y) {

			// Discard well if it is not interesting or filtered
			if (is_filtered(x,y))
				continue;


			// Determine read class
			if (rcm_->ClassMatch(x, y, MapLibrary))
				read_class = 0;
			else if (rcm_->ClassMatch(x, y, MapTF))
				read_class = 1;
			else
				continue;

			// Gather information about key flows
			signal_sum = 0.0;
			for (int flow=0; flow<keys[read_class].flows_length()-1; ++flow) {
				if(keys[read_class][flow]>0)
					signal_sum += wells_->At(y,x,flow);
			}

			if (signal_sum < 0.3 or key_base_count[read_class] < 1)
				continue;

			key_normalizer = (float)key_base_count[read_class] / signal_sum;

			for (unsigned int flow=mChunk.flowStart; flow<mChunk.flowStart+mChunk.flowDepth; ++flow) {
				wells_->WriteFlowgram ( flow, x, y, (key_normalizer*wells_->At(y,x,flow)) );
			}

		}
	}

}
// ---------------------------------------------------------------------------------
void WellsNormalization::NormalizeKeySignal(const vector<KeySequence> & keys, const ProcessingMask* pmask){

	// Load information about the currently loaded well chunk
	WellChunk mChunk = wells_->GetChunk();
	int read_class = -1;
	float key_normalizer, signal_sum;

	vector<int> key_base_count(keys.size(), 0);
	for (unsigned int iKey=0; iKey< keys.size(); ++iKey){
		for (int flow=0; flow<keys[iKey].flows_length()-1; ++flow)
			key_base_count[iKey] += keys[iKey][flow];
	}

	// Iterate over wells in this Chunk
	for (unsigned int r=0; r< pmask->NumRows(); ++r) {
		for (unsigned int c=0; c< pmask->NumCols(); ++c) {
			int x = mChunk.colStart + c;
			int y = mChunk.rowStart + r;

			// Discard well if it is not interesting of filtered
			if (! pmask->Get(r,c))
				continue;

            // Determine read class
            if (rcm_->ClassMatch(x, y, MapLibrary))
                read_class = 0;
            else if (rcm_->ClassMatch(x, y, MapTF))
                read_class = 1;
            else
                continue;

			// Gather information about key flows
			signal_sum = 0.0;
			for (int flow=0; flow<keys[read_class].flows_length()-1; ++flow) {
				if(keys[read_class][flow]>0)
					signal_sum += wells_->At(y,x,flow);
			}

			if (signal_sum < 0.3 or key_base_count[read_class] < 1)
				continue;

			key_normalizer = (float)key_base_count[read_class] / signal_sum;

			for (unsigned int flow=mChunk.flowStart; flow<mChunk.flowStart+mChunk.flowDepth; ++flow) {
				wells_->WriteFlowgram ( flow, x, y, (key_normalizer*wells_->At(y,x,flow)) );
			}

		}
	}

}
// ---------------------------------------------------------------------------------
void WellsNormalization::CorrectSignalBias(const vector<KeySequence> & keys){
	if (is_disabled_ or ! doSignalBiasCorr_)
		return;
	// assume 1.wells key normalized.  DoKeyNormalization must run first

	// POTENTIAL IMPROVEMENTS ....
	// good mask should only consider 0 and 1 when calculating residuals
	// subtract zero range should be after keys

	// KEEP in mind:
	// cannot in sync in bg region.  need to remove inidivual well difference.

	ClockTimer timer;
	timer.StartTimer();
	// processing parameters
	unsigned int numMem = 10;
	unsigned int flowZeroStart = 0, flowZeroEnd = 50; // parameters of SubtractInvdividualWellZero
	unsigned int goodFlowStart = 0, goodFlowEnd = 30; // first and last flows used in finding good mask
	double goodResidualThreshold = 0.16;  // threshold for good wells
	unsigned int winEachSide = 16;
	unsigned int startFlowOffset = 25;
	unsigned int nucOffsetStartFlow = 0;
	unsigned int nucOffsetEndFlow = 320;
	unsigned int nucStrengthStartFlow = 25;
	unsigned int nucStrengthEndFlow = 25+32;
	unsigned int minNumWells = 300;
	bool useResidualMeanAsThreshold = true;
	//bool pinZero = true;




	// initialize masks
	WellChunk mChunk = wells_->GetChunk();
	maskAll_ = new ProcessingMask(mChunk.rowHeight, mChunk.colWidth);  // mask of all unfiltered wells
	maskGood_ = new ProcessingMask(mChunk.rowHeight, mChunk.colWidth); // mask of good quality wells.  subset of unfiltered wells
	if (DEBUG_ ){
		printf("rowStart %zu, rowHeight %zu, colStart %zu, colWidth %zu, flowStart %zu, flowDepth %zu\n", mChunk.rowStart, mChunk.rowHeight, mChunk.colStart, mChunk.colWidth, mChunk.flowStart, mChunk.flowDepth);
	}


	// pre-correction
	UpdateMaskAll();
	NormalizeKeySignal(keys, maskAll_);
	Update01(numMem, goodFlowStart, goodFlowEnd);
	SubtractInvdividualWellZero(flowZeroStart, flowZeroEnd);
if (DEBUG_ && (mChunk.rowStart==50 && mChunk.colStart==100)){
		printf("Zero subtracted (%d,%d): avg= %1.5f, min=%1.5f, max=%1.5f, #subtracted=%d\n",zeroFlowStartUsed_, zeroFlowStopUsed_, avgZeroSubtracted_, minZeroSubtracted_, maxZeroSubtracted_, numWellsSubtractedZero_);
	}
	NormalizeKeySignal(keys, maskAll_);
	Update01(numMem, goodFlowStart, goodFlowEnd);
	UpdateMaskGood(goodResidualThreshold, useResidualMeanAsThreshold);
	if (DEBUG_ && (mChunk.rowStart==0 && mChunk.colStart==100)){
		printf("# passfilter, good wells are %d, %d\n", numPassFilter_, numGood_);
	}
	if (numPassFilter_ <minNumWells || numGood_ < minNumWells){
		return;
	}

	//BalanceNucStrength(nucStrengthStartFlow, nucStrengthEndFlow);

	if (DEBUG_ && (mChunk.rowStart==50 && mChunk.colStart==100)){
		printf("nucStrength T, A, C, G = %1.3f, %1.3f, %1.3f, %1.3f\n",nucStrengthT_, nucStrengthA_, nucStrengthC_, nucStrengthG_) ;

	}

	vector<float> sig0, sig1, fract0, fract1;
	Find01(sig0, sig1, fract0, fract1);
	vector<float> sig0_orig(sig0), sig1_orig(sig1), fract0_orig(fract0), fract1_orig(fract1);  // save a copy
	CorrectFlowOffset(sig0, fract0, sig1, fract1, winEachSide, startFlowOffset, maskAll_, flowCorrectMethod_); // is weighted avg really right?  0-mer fluctuating
	Find01(sig0, sig1, fract0, fract1);
	CorrectNucOffset(sig0, fract0, nucOffsetStartFlow, nucOffsetEndFlow, maskAll_);



	Update01(numMem, goodFlowStart, goodFlowEnd);
	Find01(sig0, sig1, fract0, fract1);
	SubtractInvdividualWellZero(flowZeroStart, flowZeroEnd);
	NormalizeKeySignal(keys, maskAll_);
	//BalanceNucStrength(nucStrengthStartFlow, nucStrengthEndFlow);



	if (DEBUG_ && (1)){
		Find01(sig0, sig1, fract0, fract1);
		char nuc [] = "TACG";
		for (int n=0; n <3; n++){
			for (unsigned int f =0; f<sig0.size();f++){
				if (flow_order_->operator [](f) == nuc[n]){
					//printf("%c: f, offset0, sig0Target, %d, %1.3f, %1.3f, orig: %1.3f, %1.3f, %1.3f, %1.3f, correct: %1.3f, %1.3f, %1.3f, %1.3f\n", nuc[n], f, offset0_[f], sig0Target_[f], sig0_orig[f], fract0_orig[f]*100, sig1_orig[f], fract1_orig[f]*100, sig0[f], fract0[f]*100, sig1[f], fract1[f]*100);
					printf("%c: f,  %d, orig: %1.3f, %1.3f, %1.3f, %1.3f, correct: %1.3f, %1.3f, %1.3f, %1.3f, target: %1.3f, %1.3f\n", nuc[n], f, sig0_orig[f], fract0_orig[f]*100, sig1_orig[f], fract1_orig[f]*100, sig0[f], fract0[f]*100, sig1[f], fract1[f]*100, sig0Target_[f], sig1Target_[f]);
				}
			}
		}
		printf("nucOffset T,A,C,G, avg %1.4f, %1.4f, %1.4f, %1.4f, %1.4f\n", offsetT_, offsetA_, offsetC_, offsetG_, offsetAvg_);
		printf("nucOffsetStartFlowUsed, nucOffsetEndFlowUsed, %d, %d\n", nucOffsetStartFlowUsed_, nucOffsetEndFlowUsed_);
		printf("nucStrength T, A, C, G = %1.3f, %1.3f, %1.3f, %1.3f\n",nucStrengthT_, nucStrengthA_, nucStrengthC_, nucStrengthG_) ;

	}


	delete maskAll_;
	delete maskGood_;
	DeleteMark01Data();
}
// ---------------------------------------------------------------------------------
void WellsNormalization::BalanceNucStrength(unsigned int nucStrengthStartFlow, unsigned int nucStrengthEndFlow){
	double nucStrengthT = 0., nucStrengthA = 0., nucStrengthC = 0., nucStrengthG = 0.;
	unsigned int countT =0, countA = 0, countC=0, countG = 0;

	WellChunk mChunk = wells_->GetChunk();
	nucStrengthStartFlow = max(nucStrengthStartFlow, (unsigned int) mChunk.flowStart);
	nucStrengthEndFlow = min(nucStrengthEndFlow, (unsigned int) (mChunk.flowStart+mChunk.flowDepth));


	const unsigned int colStart = mChunk.colStart;
	const unsigned int rowStart = mChunk.rowStart;


	// calculate mean 1-mer intensity
	for (unsigned int flow = nucStrengthStartFlow; flow < nucStrengthEndFlow; flow++){
		for (unsigned int r =0; r<maskGood_->NumRows(); r++){
			for (unsigned int c=0; c<maskGood_->NumCols(); c++){
				const unsigned int y = rowStart + r;
				const unsigned int x = colStart + c;
				if (m01data_->category(y, x, flow) == 1 && maskGood_->Get(r,c)){
					switch (flow_order_->operator [](flow)){
					case 'T':
						nucStrengthT += wells_->At(y,x,flow);
						countT++;
						break;
					case 'A':
						nucStrengthA += wells_->At(y,x,flow);
						countA++;
						break;
					case 'C':
						nucStrengthC += wells_->At(y,x,flow);
						countC++;
						break;
					case 'G':
						nucStrengthG += wells_->At(y,x,flow);
						countG++;
						break;
					default:
						ION_ABORT("Unknown flow character.");
					}
				}
			}
		}
	}
	nucStrengthT /= (double) countT;
	nucStrengthA /= (double) countA;
	nucStrengthC /= (double) countC;
	nucStrengthG /= (double) countG;

	// save a copy
	nucStrengthT_ = nucStrengthT;
	nucStrengthA_ = nucStrengthA;
	nucStrengthC_ = nucStrengthC;
	nucStrengthG_ = nucStrengthG;

	nucStrengthT /= nucStrengthG;
	nucStrengthA /= nucStrengthG;
	nucStrengthC /= nucStrengthG;



	// apply correction
	for (unsigned int flow = mChunk.flowStart; flow < mChunk.flowStart+mChunk.flowDepth;flow++ ){
		float normFactor = 1.f;
		switch (flow_order_->operator [](flow)){
		case 'T':
			normFactor = 1.f/nucStrengthT;
			break;
		case 'A':
			normFactor = 1.f/nucStrengthA;
			break;
		case 'C':
			normFactor = 1.f/nucStrengthC;
			break;
		case 'G':
			continue;
		default:
			ION_ABORT("Unknown flow character.");
		}
		for (unsigned int r =0; r<maskAll_->NumRows(); r++){
			for (unsigned int c=0; c<maskAll_->NumCols(); c++){
				const unsigned int y = rowStart + r;
				const unsigned int x = colStart + c;
				if (maskAll_->Get(r,c)){
					wells_->WriteFlowgram ( flow, x, y, (normFactor*wells_->At(y,x,flow)) );
				}
			}
		}
	}









}


// ---------------------------------------------------------------------------------
void WellsNormalization::CorrectNucOffset(const vector<float> sig0, const vector<float> fract0, unsigned int nucOffsetStartFlow, unsigned int nucOffsetEndFlow, ProcessingMask * pmask){
	WellChunk mChunk = wells_->GetChunk();
	nucOffsetStartFlow = max(nucOffsetStartFlow, (unsigned int) mChunk.flowStart);
	nucOffsetEndFlow = min(nucOffsetEndFlow, (unsigned int) (mChunk.flowStart+mChunk.flowDepth));

	double offsetT = 0., offsetA=0., offsetC=0., offsetG = 0.;
	double normT = 0., normA = 0., normC = 0., normG = 0.;

	// calculate per-nuc offset
	for (unsigned int flow = nucOffsetStartFlow; flow < nucOffsetEndFlow; flow++){
		const unsigned int f = flow - mChunk.flowStart;  // flow is indexed as in seq run, f is indexed offset by flowStart
		if (flow_order_->operator [](flow) == 'T'){
			offsetT += sig0[f]*fract0[f];
			normT += fract0[f];
		} else if (flow_order_->operator [](flow) == 'A'){
			offsetA += sig0[f]*fract0[f];
			normA += fract0[f];
		} else if (flow_order_->operator [](flow) == 'C'){
			offsetC += sig0[f]*fract0[f];
			normC += fract0[f];
		} else if (flow_order_->operator [](flow) == 'G'){
			offsetG += sig0[f]*fract0[f];
			normG += fract0[f];
		}

	}
	if (normT > 0.){
		offsetT /= normT;
	}
	if (normA > 0.){
		offsetA /= normA;
	}
	if (normC > 0.){
		offsetC /= normC;
	}
	if (normG > 0.){
		offsetG /= normG;
	}

	double offsetAvg = 0.25*(offsetT + offsetA + offsetC + offsetG);

	// apply correction factor
	for (unsigned int r=0; r<pmask->NumRows(); r++){
		for (unsigned int c=0; c<pmask->NumCols(); c++){
			if (pmask->Get(r,c)){
				int x = mChunk.colStart + c;
				int y = mChunk.rowStart + r;
				for (unsigned int flow=mChunk.flowStart; flow < mChunk.flowStart+mChunk.flowDepth; flow++){
					const unsigned int f = flow - mChunk.flowStart;  // flow is indexed as in seq run, f is indexed offset by flowStart
					double offset = 0.;
					switch (flow_order_->operator [](flow)){
					case 'T':
						offset = offsetT;
						break;
					case 'A':
						offset = offsetA;
						break;
					case 'C':
						offset = offsetC;
						break;
					case 'G':
						offset = offsetG;
						break;
					default:
						ION_ABORT("Unknown flow character.");

					}
					wells_->WriteFlowgram ( flow, x, y, (wells_->At(y,x,flow)  - offset +offsetAvg ) );
				}

			}
		}
	}

	// save a copy
	offsetT_ = offsetT;
	offsetA_ = offsetA;
	offsetC_ = offsetC;
	offsetG_ = offsetG;
	offsetAvg_ = offsetAvg;
	nucOffsetStartFlowUsed_ =nucOffsetStartFlow;
	nucOffsetEndFlowUsed_ =nucOffsetEndFlow;

}

// ---------------------------------------------------------------------------------
void WellsNormalization::CorrectFlowOffset(const vector<float> sig0, const vector<float> fract0, const vector<float> sig1, const vector<float> fract1, const unsigned int winEachSide, const unsigned int startFlow, const ProcessingMask* pmask, string method){
	WellChunk mChunk = wells_->GetChunk();
	unsigned flowStart = mChunk.flowStart;
	sig0Target_.resize(sig0.size(), 0.);
	sig1Target_.resize(sig0.size(), 0.);
	offsetFactor_.resize(sig0.size(), 0.);
	scaleFactor_.resize(sig0.size(), 0.);
	unsigned int numFlows = sig0.size();

	// find target 0-mer and 1-mer signals
	for (unsigned int f=startFlow; f < numFlows; f++){
		// target 0-mer
		double norm = 0.;
		sig0Target_[f] = 0.;
		for (unsigned int fw = max(flowStart, f-winEachSide); fw < min(numFlows, f+winEachSide); fw++){
			if (flow_order_->operator [](f) == flow_order_->operator [](fw)){  // per-nuc offset
				sig0Target_[f] += sig0[fw] * fract0[fw];
				norm += fract0[fw];
			}
		}
		if (norm >0.){
			sig0Target_[f] /= norm;
		}

		// target 1-mer
		norm = 0.;
		sig1Target_[f] = 0.f;
		for (unsigned int fw = max(flowStart, f-winEachSide); fw < min(numFlows, f+winEachSide); fw++){
			if (flow_order_->operator [](f) == flow_order_->operator [](fw)){  // per-nuc offset
				sig1Target_[f] += sig1[fw] * fract1[fw];
				norm += fract1[fw];
			}
		}
		if (norm >0.){
			sig1Target_[f] /= norm;
		}
	}


	// apply correction
	if (method == "pinZero"){
		// Shift each flow 0-mer as 0.
		for (unsigned int f=startFlow; f < numFlows; f++){
			offsetFactor_[f] = -sig0[f];
		}

		// apply correction factors
		for (unsigned int r=0; r<pmask->NumRows(); r++){
			for (unsigned int c=0; c<pmask->NumCols(); c++){
				if (pmask->Get(r,c)){
					int x = mChunk.colStart + c;
					int y = mChunk.rowStart + r;
					for (unsigned int f=startFlow; f < numFlows; f++){
						wells_->WriteFlowgram ( f, x, y, (wells_->At(y,x,f)  + offsetFactor_[f] ) );
					}

				}
			}
		}


	} else if (method == "fix01"){
	    float minSep = 0.2;  // minimium zero and 1-mer separation for correction.
	    // transformation: s -> f s + d
	    // solve:
	    // t1 = f s1 + d
	    // t0 = f s0 + d
	    //
		//f = (t1-t0)/(s1-s0)
	    //d = (t0*s1-t1*s0)/(s1-s0)
		//if s < s1: s -> f s + d
		//if s >= s1: s -> s +t1-s1
		for (unsigned int f=startFlow; f < numFlows; f++){
			sig1Target_[f] = sig1[f];
			if ( (sig1[f] - sig0[f]) > minSep){
				scaleFactor_[f] = (sig1Target_[f] - sig0Target_[f])/(sig1[f] - sig0[f]);
				offsetFactor_[f] = (sig0Target_[f]*sig1[f]-sig1Target_[f]*sig0[f])/(sig1[f] - sig0[f]);
			} else {
				scaleFactor_[f] = 0.;
				offsetFactor_[f] = 0.;
			}
		}

		// apply correction factors
		for (unsigned int f=startFlow; f < numFlows; f++){
			if (scaleFactor_[f] == 0. && offsetFactor_[f] == 0.)
				continue;
			for (unsigned int r=0; r<pmask->NumRows(); r++){
				for (unsigned int c=0; c<pmask->NumCols(); c++){
					if (pmask->Get(r,c)){
						int x = mChunk.colStart + c;
						int y = mChunk.rowStart + r;
						float s = (wells_->At(y,x,f));
						s = s<sig1[f]?scaleFactor_[f]*s + offsetFactor_[f] : s + sig1Target_[f]-sig1[f];
						wells_->WriteFlowgram ( f, x, y,  s);
					}

				}
			}
		}

	}
	else {
		// default method: Shift each flow 0-mer to target 0-mer
		for (unsigned int f=startFlow; f < numFlows; f++){
			offsetFactor_[f] = -sig0[f] + sig0Target_[f];
		}

		// apply correction factors
		for (unsigned int r=0; r<pmask->NumRows(); r++){
			for (unsigned int c=0; c<pmask->NumCols(); c++){
				if (pmask->Get(r,c)){
					int x = mChunk.colStart + c;
					int y = mChunk.rowStart + r;
					for (unsigned int f=startFlow; f < numFlows; f++){
						wells_->WriteFlowgram ( f, x, y, (wells_->At(y,x,f)  + offsetFactor_[f] ) );
					}

				}
			}
		}

	}




}
// ---------------------------------------------------------------------------------
void WellsNormalization::DeleteMark01Data(){
	delete m01data_;
	m01data_ = NULL;

}

// ---------------------------------------------------------------------------------
unsigned int WellsNormalization::UpdateMaskAll(){
	WellChunk mChunk = wells_->GetChunk();

	int numPassFilter=0;
	for (unsigned int r =0; r<maskAll_->NumRows(); r++){
		for (unsigned int c=0; c<maskAll_->NumCols(); c++){
			int x = mChunk.colStart + c;
			int y = mChunk.rowStart + r;
			bool passFilter = !is_filtered_libOnly(x,y);
			maskAll_->Set(r,c,  passFilter);
			if (passFilter){
				numPassFilter++;
			}
		}
	}

	//printf("number pass filter is %d\n", numPassFilter);

	return numPassFilter;
}
// ---------------------------------------------------------------------------------
unsigned int WellsNormalization::UpdateMaskGood(const double goodResidualThreshold, const bool useResidualMeanAsThreshold){
	WellChunk mChunk = wells_->GetChunk();
	const unsigned int rowStart = mChunk.rowStart;
	const unsigned int colStart = mChunk.colStart;
	const unsigned int flowStart = mChunk.flowStart;

	double threshold = goodResidualThreshold;
	if (useResidualMeanAsThreshold){
		double sum = 0.;
		int count = 0;
		for (unsigned int r =0; r<maskAll_->NumRows(); r++){
			for (unsigned int c=0; c<maskAll_->NumCols(); c++){
				if (maskAll_->Get(r,c)){
					sum += m01data_->residual(rowStart+r,colStart+c);
					count ++;
				}
			}
		}
		if (count > 0){
			threshold = sum/(double) count;
		}
	}


	int numGood=0;
	for (unsigned int r =0; r<maskAll_->NumRows(); r++){
		for (unsigned int c=0; c<maskAll_->NumCols(); c++){
			maskGood_->Set(r,c,false);

			if (maskAll_->Get(r,c)){
				// calculate average residual from classification
				if (m01data_->residual(rowStart+r,colStart+c)<= threshold){
					numGood++;
					maskGood_->Set(r,c,true);
				}
			}

		}
	}

	//printf("number pass filter is %d\n", numPassFilter);

	numGood_ = numGood;
	return numGood;
}






// ---------------------------------------------------------------------------------

unsigned int WellsNormalization::Update01(const unsigned int numMem, const unsigned int goodFlowStart, const unsigned int goodFlowEnd){
	WellChunk mChunk = wells_->GetChunk();
	int numPassFilter;

	if (! m01data_){
		m01data_ = new Mark01Data(mChunk.rowStart, mChunk.rowHeight, mChunk.colStart, mChunk.colWidth, mChunk.flowStart, mChunk.flowDepth, wells_, rcm_);
	}
	m01data_->Initialize(mChunk.rowStart, mChunk.rowHeight, mChunk.colStart, mChunk.colWidth, mChunk.flowStart, mChunk.flowDepth, wells_, rcm_);
	numPassFilter = m01data_->Classify(maskAll_, numMem, goodFlowStart, goodFlowEnd);
	numPassFilter_ = numPassFilter;
	return numPassFilter;

}


// ---------------------------------------------------------------------------------
double WellsNormalization::SubtractInvdividualWellZero(const size_t zeroFlowStart, const size_t zeroFlowStop){
	WellChunk mChunk = wells_->GetChunk();
	const size_t zeroFlowStartUsed = max(zeroFlowStart, mChunk.flowStart);
	const size_t zeroFlowStopUsed = min(zeroFlowStop, mChunk.flowStart+mChunk.flowDepth);
	double avgZeroSubtracted = 0., minZeroSubtracted = 1000, maxZeroSubtracted = -1000;
	unsigned int countSubtracted = 0;
	for (unsigned int r = mChunk.rowStart; r < mChunk.rowStart+mChunk.rowHeight; r++){
		for (unsigned int c = mChunk.colStart; c < mChunk.colStart+mChunk.colWidth; c++){
			if (!maskAll_->Get(r-mChunk.rowStart, c - mChunk.colStart))
				continue;
			// find average zero
			double avgZero=0.;
			unsigned int zeroCount = 0;
			for (unsigned int f = zeroFlowStartUsed; f < zeroFlowStopUsed; f++){
				if (m01data_->category(r,c,f) == 0){
					avgZero += (double) wells_->At(r, c, f);
					zeroCount++;
				}
			}

			// subtract average zero
			if (zeroCount >0){
				avgZero /= (double) zeroCount;
				for (unsigned int f = mChunk.flowStart; f < mChunk.flowStart+mChunk.flowDepth; f++){
					wells_->WriteFlowgram ( f, c, r, (wells_->At(r,c,f) -avgZero ) );
				}
				avgZeroSubtracted += avgZero;
				countSubtracted ++;
				if (avgZero<minZeroSubtracted)
					minZeroSubtracted = avgZero;
				if (avgZero>maxZeroSubtracted)
					maxZeroSubtracted = avgZero;
			}

		}
	}
	if (countSubtracted > 0)
		avgZeroSubtracted/=(double) countSubtracted;

	// save a copy
	zeroFlowStartUsed_ = zeroFlowStartUsed;
	zeroFlowStopUsed_ = zeroFlowStopUsed;
	numWellsSubtractedZero_ = countSubtracted;
	avgZeroSubtracted_ = avgZeroSubtracted;
	minZeroSubtracted_ = minZeroSubtracted;
	maxZeroSubtracted_ = maxZeroSubtracted;


	return avgZeroSubtracted;
}

// ---------------------------------------------------------------------------------
void WellsNormalization::Find01(vector<float>& sig0, vector<float>& sig1, vector<float>& fract0, vector<float>& fract1){
	// Load information about the currently loaded well chunk
	WellChunk mChunk = wells_->GetChunk();
	// classify flows as 0,1 or 2 at individual wells

	sig0.resize(mChunk.flowDepth);
	sig1.resize(mChunk.flowDepth);
	fract0.resize(mChunk.flowDepth);
	fract1.resize(mChunk.flowDepth);
	for (unsigned int f = 0; f < mChunk.flowDepth; f++){
		unsigned int flow = mChunk.flowStart + f;
		sig0[f] = m01data_->MeanSignal(0, flow, maskGood_);
		sig1[f] = m01data_->MeanSignal(1, flow, maskGood_);
		fract0[f] = m01data_->Fraction(0, flow, maskGood_);
		fract1[f] = m01data_->Fraction(1, flow, maskGood_);
		//printf("flow, sig0, sig1, fract0, fract1 = %d, %1.2f, %1.2f, %1.2f, %1.2f\n", flow, sig0[f], sig1[f], fract0[f], fract1[f]);
	}










}


// =================================================================================
