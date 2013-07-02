#include "BandedSmithWaterman.h"

// define our static constants
const float CBandedSmithWaterman::FLOAT_NEGATIVE_INFINITY = (float)-1e+30;

const DirectionType CBandedSmithWaterman::Directions_STOP     = 0;
const DirectionType CBandedSmithWaterman::Directions_LEFT     = 1;
const DirectionType CBandedSmithWaterman::Directions_DIAGONAL = 2;
const DirectionType CBandedSmithWaterman::Directions_UP       = 3;

const PositionType CBandedSmithWaterman::Position_REF_AND_QUERY_ZERO    = 0;
const PositionType CBandedSmithWaterman::Position_REF_ZERO              = 1;
const PositionType CBandedSmithWaterman::Position_QUERY_ZERO            = 2;
const PositionType CBandedSmithWaterman::Position_REF_AND_QUERO_NONZERO = 3;

// constructor
CBandedSmithWaterman::CBandedSmithWaterman(float matchScore, float mismatchScore, float gapOpenPenalty, float gapExtendPenalty, unsigned int bandWidth) 
: mCurrentMatrixSize(0)
, mCurrentAnchorSize(0)
, mCurrentAQSumSize(0)
, mBandwidth(bandWidth)
, mPointers(NULL)
, mMatchScore(matchScore)
, mMismatchScore(mismatchScore)
, mGapOpenPenalty(gapOpenPenalty)
, mGapExtendPenalty(gapExtendPenalty)
, mAnchorGapScores(NULL)
, mBestScores(NULL)
, mReversedAnchor(NULL)
, mReversedQuery(NULL)
, mUseHomoPolymerGapOpenPenalty(false)
{
	CreateScoringMatrix();

	//if((bandWidth % 2) != 1) {
		//printf("ERROR: The bandwidth must be an odd number.\n");
		//exit(1);
	//}

	try {
		mBestScores	 = new float[bandWidth + 2];
		mAnchorGapScores = new float[bandWidth + 2];
	} catch(bad_alloc) {
		printf("ERROR: Unable to allocate enough memory for the banded Smith-Waterman algorithm.\n");
		exit(1);
	}
}

// destructor
CBandedSmithWaterman::~CBandedSmithWaterman(void) {
	if(mPointers)              delete [] mPointers;
	if(mAnchorGapScores)       delete [] mAnchorGapScores;
	if(mBestScores)            delete [] mBestScores;
	if(mReversedAnchor)        delete [] mReversedAnchor;
	if(mReversedQuery)         delete [] mReversedQuery;
}

// aligns the query sequence to the anchor using the Smith Waterman Gotoh algorithm
void CBandedSmithWaterman::Align(unsigned int& referenceAl, string& cigarAl, const string& s1, const string& s2, pair< pair<unsigned int, unsigned int>, pair<unsigned int, unsigned int> >& hr) {


	
	unsigned int rowStart = min(hr.first.first, (unsigned int)hr.second.first);
	hr.first.first    -= rowStart;
	hr.second.first   -= rowStart;
	
	//bool isLegalBandWidth = (s2.length() - hr.QueryBegin) > (mBandwidth / 2);
	//     isLegalBandWidth = isLegalBandWidth && ((s1.length() - hr.Begin) > (mBandwidth / 2));



	// check the lengths of the input sequences
	//if( (s1.length() <= 0) || (s2.length() <= 0) || (s1.length() < s2.length()) ) {
	//	printf("ERROR: An unexpected sequence length was encountered during pairwise alignment.\n");
	//	printf("Sequence lengths are listed as following:\n");
	//	printf("1. Reference length: %u\n2. Query length: %u\n", s1.length(), s2.length());
		//printf("3. Hash region in reference:%4u-%4u\n", hr.Begin + rowStart, hr.End);
		//printf("4. Hash region in query:    %4u-%4u\n", hr.QueryBegin + rowStart, hr.QueryEnd);
	//	exit(1);
	//}

	
	// determine the hash region type
	unsigned int rowOffset;
	unsigned int columnOffset;
	PositionType positionType;

	if(hr.first.first == 0) {
		if(hr.second.first == 0) {
			rowOffset    = 1;
			columnOffset = (mBandwidth / 2) + 1;
			positionType = Position_REF_AND_QUERY_ZERO;
		} else {
			rowOffset    = 1 - hr.second.first;
			columnOffset = (mBandwidth / 2) + 1 + hr.second.first;
			positionType = Position_REF_ZERO;
		}
	} else {
		if(hr.second.first == 0) {
			rowOffset    = 1;
			columnOffset = (mBandwidth / 2) + 1 - hr.first.first;
			positionType = Position_QUERY_ZERO;
		} else {
			rowOffset    = 1 - hr.second.first;
			columnOffset = (mBandwidth / 2) + 1 + hr.second.first - hr.first.first;
			positionType = Position_REF_AND_QUERO_NONZERO;
		}
	}

	// =========================
	// Reinitialize the matrices
	// =========================
	
	ReinitializeMatrices(positionType, s1.length(), s2.length(), hr);

	// =======================================
	// Banded Smith-Waterman forward algorithm
	// =======================================

	unsigned int bestColumn	= 0;
	unsigned int bestRow	= 0;
	float bestScore         = FLOAT_NEGATIVE_INFINITY;
	float currentQueryGapScore;

	// rowNum and column indicate the row and column numbers in the Smith-Waterman matrix respectively
	unsigned int rowNum    = hr.second.first;
	unsigned int columnNum = hr.first.first;

	// indicates how many rows including blank elements in the Banded SmithWaterman
	int numBlankElements = (mBandwidth / 2) - columnNum;

	//cout << numBlankElements << endl;
	// upper triangle matrix in Banded Smith-Waterman
	for( ; numBlankElements > 0; numBlankElements--, rowNum++){
		// in the upper triangle matrix, we always start at the 0th column
		columnNum = 0;

		// columnEnd indicates how many columns which should be dealt with in the current row
		unsigned int columnEnd = min((mBandwidth - numBlankElements), ((unsigned int) s1.length() - columnNum + 1) );
		currentQueryGapScore = FLOAT_NEGATIVE_INFINITY;
		for( unsigned int j = 0; j < columnEnd; j++){
			float score = CalculateScore(s1, s2, rowNum, columnNum, currentQueryGapScore, rowOffset, columnOffset);
			//cout << s1[columnNum] << s2[rowNum] << score << endl;
			UpdateBestScore(bestRow, bestColumn, bestScore, rowNum, columnNum, score);
			columnNum++;
		}

		// replace the columnNum to the middle column in the Smith-Waterman matrix
		columnNum = columnNum - (mBandwidth / 2);
	}
	// complete matrix in Banded Smith-Waterman
	unsigned int completeNum = min((s1.length() - columnNum - (mBandwidth / 2)), (s2.length() - rowNum));
	//cout << completeNum << endl;
	for(unsigned int i = 0; i < completeNum; i++, rowNum++){
		columnNum = columnNum - (mBandwidth / 2);

		// there are mBandwidth columns which should be dealt with in each row
		currentQueryGapScore = FLOAT_NEGATIVE_INFINITY;

		for(unsigned int j = 0; j < mBandwidth; j++){
			float score = CalculateScore(s1, s2, rowNum, columnNum, currentQueryGapScore, rowOffset, columnOffset);
			UpdateBestScore(bestRow, bestColumn, bestScore, rowNum, columnNum, score);
			//cout << s1[columnNum] << s2[rowNum] << score << endl;
			columnNum++;
		}

		// replace the columnNum to the middle column in the Smith-Waterman matrix
		// because mBandwidth is an odd number, everytime the following equation shifts a column (pluses 1).
		columnNum = columnNum - (mBandwidth / 2);
	}
	
	// lower triangle matrix
	numBlankElements = min(mBandwidth, ((unsigned int) s2.length() - rowNum));
	columnNum = columnNum - (mBandwidth / 2);
	for(unsigned int i = 0; numBlankElements > 0; i++, rowNum++, numBlankElements--) {

		mBestScores[ mBandwidth - i ] = FLOAT_NEGATIVE_INFINITY;;
		// columnEnd indicates how many columns which should be dealt with
		currentQueryGapScore = FLOAT_NEGATIVE_INFINITY;

		for( unsigned int j = columnNum; j < s1.length(); j++){
			float score = CalculateScore(s1, s2, rowNum, columnNum, currentQueryGapScore, rowOffset, columnOffset);
			UpdateBestScore(bestRow, bestColumn, bestScore, rowNum, columnNum, score);
			//cout << s1[columnNum] << s2[rowNum] << score << endl;
			columnNum++;
		}

		// replace the columnNum to the middle column in the Smith-Waterman matrix
		columnNum = columnNum - mBandwidth + i + 2;
	}

	// =========================================
	// Banded Smith-Waterman backtrace algorithm
	// =========================================

	Traceback(referenceAl, cigarAl, s1, s2, bestRow, bestColumn, rowOffset, columnOffset);

}

// calculates the score during the forward algorithm
float CBandedSmithWaterman::CalculateScore(const string& s1, const string& s2, const unsigned int rowNum, const unsigned int columnNum, float& currentQueryGapScore, const unsigned int rowOffset, const unsigned int columnOffset) {

	// initialize
	const unsigned int row      = rowNum + rowOffset;
	const unsigned int column   = columnOffset - rowNum + columnNum;
	const unsigned int position = row * (mBandwidth + 2) + column;

	// retrieve the similarity scores
	const float similarityScore      = mScoringMatrix[s1[columnNum] - 'A'][s2[rowNum] - 'A'];
	const float totalSimilarityScore = mBestScores[column] + similarityScore;

	// ================================
	// open a gap in the query sequence
	// ================================

	float queryGapExtendScore = currentQueryGapScore - mGapExtendPenalty;
	float queryGapOpenScore   = mBestScores[column - 1] - mGapOpenPenalty;

	// compute the homo-polymer gap score if enabled
	if(mUseHomoPolymerGapOpenPenalty)
		if((rowNum > 1) && (s2[rowNum] == s2[rowNum - 1]))
			queryGapOpenScore = mBestScores[column - 1] - mHomoPolymerGapOpenPenalty;

	if(queryGapExtendScore > queryGapOpenScore) {
		currentQueryGapScore = queryGapExtendScore;
		mPointers[position].mSizeOfHorizontalGaps = mPointers[position - 1].mSizeOfHorizontalGaps + 1;
	} else currentQueryGapScore = queryGapOpenScore;


	// ====================================
	// open a gap in the reference sequence
	// ====================================

	
	float anchorGapExtendScore = mAnchorGapScores[column + 1] - mGapExtendPenalty;
	float anchorGapOpenScore   = mBestScores[column + 1] - mGapOpenPenalty;

	// compute the homo-polymer gap score if enabled	
	if(mUseHomoPolymerGapOpenPenalty)
		if((columnNum > 1) && (s1[columnNum] == s1[columnNum - 1]))
			anchorGapOpenScore = mBestScores[column + 1] - mHomoPolymerGapOpenPenalty;

	if(anchorGapExtendScore > anchorGapOpenScore) {
		mAnchorGapScores[column] = anchorGapExtendScore;
		mPointers[position].mSizeOfVerticalGaps = mPointers[position - mBandwidth - 1].mSizeOfVerticalGaps + 1;
	} else mAnchorGapScores[column] = anchorGapOpenScore;

	// ======================================
	// calculate the best score and direction
	// ======================================

	//mBestScores[column] = MaxFloats(totalSimilarityScore, mAnchorGapScores[column], currentQueryGapScore);
	mBestScores[column] = MaxFloats(totalSimilarityScore, currentQueryGapScore, mAnchorGapScores[column]);

	// determine the traceback direction
	// diagonal (445364713) > stop (238960195) > up (214378647) > left (166504495)
	if(mBestScores[column] == 0)                            mPointers[position].Direction = Directions_STOP;
	else if(mBestScores[column] == totalSimilarityScore)    mPointers[position].Direction = Directions_UP;
	else if(mBestScores[column] == currentQueryGapScore)   mPointers[position].Direction = Directions_LEFT;
	else                                                    mPointers[position].Direction = Directions_DIAGONAL;

	return mBestScores[column];
}

// corrects the homopolymer gap order for forward alignments
void CBandedSmithWaterman::CorrectHomopolymerGapOrder(const unsigned int numBases, const unsigned int numMismatches) {


	// this is only required for alignments with mismatches
	//if(al.NumMismatches == 0) return;
	if ( numMismatches == 0 ) return;

	// localize the alignment data
	//char* pReference = al.Reference.Data();
	//char* pQuery     = al.Query.Data();
	//const unsigned int numBases = al.Reference.Length();
	char* pReference = mReversedAnchor;
	char* pQuery     = mReversedQuery;

	// initialize
	bool hasReferenceGap = false, hasQueryGap = false;
	char* pNonGapSeq = NULL;
	char* pGapSeq    = NULL;
	char nonGapBase  = 'J';

	// identify gapped regions
	for(unsigned int i = 0; i < numBases; i++) {

		// check if the current position is gapped
		hasReferenceGap = false;
		hasQueryGap     = false;

		if(pReference[i] == GAP) {
			hasReferenceGap = true;
			pNonGapSeq      = pQuery;
			pGapSeq         = pReference;
			nonGapBase      = pQuery[i];
		}

		if(pQuery[i] == GAP) {
			hasQueryGap = true;
			pNonGapSeq  = pReference;
			pGapSeq     = pQuery;
			nonGapBase  = pReference[i];
		}

		// continue if we don't have any gaps
		if(!hasReferenceGap && !hasQueryGap) continue;

		// sanity check
		if(hasReferenceGap && hasQueryGap) {
			printf("ERROR: Found a gap in both the reference sequence and query sequence.\n");
			exit(1);
		}

		// find the non-gapped length (forward)
		unsigned short numGappedBases = 0;
		unsigned short nonGapLength   = 0;
		unsigned short testPos = i;
		while(testPos < numBases) {

			const char gs  = pGapSeq[testPos];
			const char ngs = pNonGapSeq[testPos];

			bool isPartofHomopolymer = false;
			if(((gs == nonGapBase) || (gs == GAP)) && (ngs == nonGapBase)) isPartofHomopolymer = true;
			if(!isPartofHomopolymer) break;

			if(gs == GAP) numGappedBases++;
			else nonGapLength++;
			testPos++;
		}

		// fix the gap order
		if(numGappedBases != 0) {
			char* pCurrentSequence = pGapSeq + i;
			memset(pCurrentSequence, nonGapBase, nonGapLength);
			pCurrentSequence += nonGapLength;
			memset(pCurrentSequence, GAP, numGappedBases);
		}

		// increment
		i += numGappedBases + nonGapLength - 1;
	}
}

// creates a simple scoring matrix to align the nucleotides and the ambiguity code N
void CBandedSmithWaterman::CreateScoringMatrix(void) {

	unsigned int nIndex = 13;
	unsigned int xIndex = 23;

	// define the N score to be 1/4 of the span between mismatch and match
	//const short nScore = mMismatchScore + (short)(((mMatchScore - mMismatchScore) / 4.0) + 0.5);

	// calculate the scoring matrix
	for(unsigned char i = 0; i < MOSAIK_NUM_NUCLEOTIDES; i++) {
		for(unsigned char j = 0; j < MOSAIK_NUM_NUCLEOTIDES; j++) {

			// N.B. matching N to everything (while conceptually correct) leads to some
			// bad alignments, lets make N be a mismatch instead.

			// add the matches or mismatches to the hashtable (N is a mismatch)
			if((i == nIndex) || (j == nIndex)) mScoringMatrix[i][j] = mMismatchScore;
			else if((i == xIndex) || (j == xIndex)) mScoringMatrix[i][j] = mMismatchScore;
			else if(i == j) mScoringMatrix[i][j] = mMatchScore;
			else mScoringMatrix[i][j] = mMismatchScore;
		}
	}

	// add ambiguity codes
	mScoringMatrix['M' - 'A']['A' - 'A'] = mMatchScore;	// M - A
	mScoringMatrix['A' - 'A']['M' - 'A'] = mMatchScore;
	// add ambiguity codes
	mScoringMatrix['M' - 'A']['A' - 'A'] = mMatchScore;	// M - A
	mScoringMatrix['A' - 'A']['M' - 'A'] = mMatchScore;
	mScoringMatrix['M' - 'A']['C' - 'A'] = mMatchScore; // M - C
	mScoringMatrix['C' - 'A']['M' - 'A'] = mMatchScore;

	mScoringMatrix['R' - 'A']['A' - 'A'] = mMatchScore;	// R - A
	mScoringMatrix['A' - 'A']['R' - 'A'] = mMatchScore;
	mScoringMatrix['R' - 'A']['G' - 'A'] = mMatchScore; // R - G
	mScoringMatrix['G' - 'A']['R' - 'A'] = mMatchScore;

	mScoringMatrix['W' - 'A']['A' - 'A'] = mMatchScore;	// W - A
	mScoringMatrix['A' - 'A']['W' - 'A'] = mMatchScore;
	mScoringMatrix['W' - 'A']['T' - 'A'] = mMatchScore; // W - T
	mScoringMatrix['T' - 'A']['W' - 'A'] = mMatchScore;

	mScoringMatrix['S' - 'A']['C' - 'A'] = mMatchScore;	// S - C
	mScoringMatrix['C' - 'A']['S' - 'A'] = mMatchScore;
	mScoringMatrix['S' - 'A']['G' - 'A'] = mMatchScore; // S - G
	mScoringMatrix['G' - 'A']['S' - 'A'] = mMatchScore;

	mScoringMatrix['Y' - 'A']['C' - 'A'] = mMatchScore;	// Y - C
	mScoringMatrix['C' - 'A']['Y' - 'A'] = mMatchScore;
	mScoringMatrix['Y' - 'A']['T' - 'A'] = mMatchScore; // Y - T
	mScoringMatrix['T' - 'A']['Y' - 'A'] = mMatchScore;

	mScoringMatrix['K' - 'A']['G' - 'A'] = mMatchScore;	// K - G
	mScoringMatrix['G' - 'A']['K' - 'A'] = mMatchScore;
	mScoringMatrix['K' - 'A']['T' - 'A'] = mMatchScore; // K - T
	mScoringMatrix['T' - 'A']['K' - 'A'] = mMatchScore;

	mScoringMatrix['V' - 'A']['A' - 'A'] = mMatchScore;	// V - A
	mScoringMatrix['A' - 'A']['V' - 'A'] = mMatchScore;
	mScoringMatrix['V' - 'A']['C' - 'A'] = mMatchScore; // V - C
	mScoringMatrix['C' - 'A']['V' - 'A'] = mMatchScore;
	mScoringMatrix['V' - 'A']['G' - 'A'] = mMatchScore; // V - G
	mScoringMatrix['G' - 'A']['V' - 'A'] = mMatchScore;

	mScoringMatrix['H' - 'A']['A' - 'A'] = mMatchScore;	// H - A
	mScoringMatrix['A' - 'A']['H' - 'A'] = mMatchScore;
	mScoringMatrix['H' - 'A']['C' - 'A'] = mMatchScore; // H - C
	mScoringMatrix['C' - 'A']['H' - 'A'] = mMatchScore;
	mScoringMatrix['H' - 'A']['T' - 'A'] = mMatchScore; // H - T
	mScoringMatrix['T' - 'A']['H' - 'A'] = mMatchScore;

	mScoringMatrix['D' - 'A']['A' - 'A'] = mMatchScore;	// D - A
	mScoringMatrix['A' - 'A']['D' - 'A'] = mMatchScore;
	mScoringMatrix['D' - 'A']['G' - 'A'] = mMatchScore; // D - G
	mScoringMatrix['G' - 'A']['D' - 'A'] = mMatchScore;
	mScoringMatrix['D' - 'A']['T' - 'A'] = mMatchScore; // D - T
	mScoringMatrix['T' - 'A']['D' - 'A'] = mMatchScore;

	mScoringMatrix['B' - 'A']['C' - 'A'] = mMatchScore;	// B - C
	mScoringMatrix['C' - 'A']['B' - 'A'] = mMatchScore;
	mScoringMatrix['B' - 'A']['G' - 'A'] = mMatchScore; // B - G
	mScoringMatrix['G' - 'A']['B' - 'A'] = mMatchScore;
	mScoringMatrix['B' - 'A']['T' - 'A'] = mMatchScore; // B - T
	mScoringMatrix['T' - 'A']['B' - 'A'] = mMatchScore;
}

// enables homo-polymer scoring
void CBandedSmithWaterman::EnableHomoPolymerGapPenalty(float hpGapOpenPenalty) {
	mUseHomoPolymerGapOpenPenalty = true;
	mHomoPolymerGapOpenPenalty    = hpGapOpenPenalty;
}

// reinitializes the matrices
void CBandedSmithWaterman::ReinitializeMatrices(const PositionType& positionType, const unsigned int& s1Length, const unsigned int& s2Length, const pair< pair<unsigned int, unsigned int>, pair<unsigned int, unsigned int> > hr) {

/*
	try {
		mBestScores	 = new float[mBandwidth + 2];
		mAnchorGapScores = new float[mBandwidth + 2];
	} catch(bad_alloc) {
		printf("ERROR: Unable to allocate enough memory for the banded Smith-Waterman algorithm.\n");
		exit(1);
	}
*/

	const unsigned int numColumns = mBandwidth + 2;
	unsigned int numRows = 0;

	switch(positionType) {
		case Position_REF_AND_QUERY_ZERO:
			numRows = s2Length + 1;
			break;
		case Position_REF_ZERO:
			numRows = s2Length - hr.second.first + 2;
			break;
		case Position_QUERY_ZERO:
			numRows = min(s2Length + 1, s1Length - hr.first.first + 2);
			break;
		case Position_REF_AND_QUERO_NONZERO:
			numRows = min(s1Length - hr.first.first + 2, s2Length - hr.second.first + 2);
			break;
	}

	// update the size of the backtrace matrix
	if((numColumns * numRows) > mCurrentMatrixSize) {

		mCurrentMatrixSize = numColumns * numRows;
		if(mPointers) delete [] mPointers;

		try {
			mPointers = new ElementInfo[mCurrentMatrixSize];
		} catch(bad_alloc) {
			printf("ERROR: Unable to allocate enough memory for the banded Smith-Waterman algorithm.\n");
			exit(1);
		}
	}

	// initialize our backtrace matrix
	ElementInfo defaultElement;
	defaultElement.Direction = Directions_STOP;
	defaultElement.mSizeOfHorizontalGaps = 1;
	defaultElement.mSizeOfVerticalGaps   = 1;

	uninitialized_fill(mPointers, mPointers + mCurrentMatrixSize, defaultElement);

	// update the sequence character arrays
	if((s1Length + s2Length) > mCurrentAQSumSize) {

		mCurrentAQSumSize = s1Length + s2Length;
		if(mReversedAnchor)	delete [] mReversedAnchor;
		if(mReversedQuery)	delete [] mReversedQuery;

		try {
			mReversedAnchor	= new char[mCurrentAQSumSize + 1]; // reversed sequence #1
			mReversedQuery	= new char[mCurrentAQSumSize + 1]; // reversed sequence #2
		} catch(bad_alloc) {
			printf("ERROR: Unable to allocate enough memory for the banded Smith-Waterman algorithm.\n");
			exit(1);
		}
	}

	// initialize the gap score and score vectors
	uninitialized_fill(mAnchorGapScores, mAnchorGapScores + mBandwidth + 2, FLOAT_NEGATIVE_INFINITY);
	memset((char*)mBestScores, 0, SIZEOF_FLOAT * (mBandwidth + 2));
	mBestScores[0]              = FLOAT_NEGATIVE_INFINITY;
	mBestScores[mBandwidth + 1] = FLOAT_NEGATIVE_INFINITY;
}

// performs the backtrace algorithm
void CBandedSmithWaterman::Traceback(unsigned int& referenceAl, string& cigarAl, const string& s1, const string& s2, unsigned int bestRow, unsigned int bestColumn, const unsigned int rowOffset, const unsigned int columnOffset){


	unsigned int currentRow		 = bestRow;
	unsigned int currentColumn	 = bestColumn;
	unsigned int currentPosition = ((currentRow + rowOffset) * (mBandwidth + 2)) + (columnOffset - currentRow + currentColumn);


	// record the numbers of row and column before the current row and column
	unsigned int previousRow	= bestRow;
	unsigned int previousColumn	= bestColumn;

	unsigned int gappedAnchorLen = 0;
	unsigned int gappedQueryLen  = 0;
	unsigned int numMismatches   = 0;

	bool keepProcessing = true;
	while(keepProcessing) {
		unsigned int nVerticalGap = 0;
		unsigned int nHorizontalGap = 0;
		switch(mPointers[currentPosition].Direction){
			case Directions_DIAGONAL:
				nVerticalGap = mPointers[currentPosition].mSizeOfVerticalGaps;
				for(unsigned int i = 0; i < nVerticalGap; i++){
					mReversedAnchor[gappedAnchorLen++] = GAP;
					mReversedQuery[gappedQueryLen++]   = s2[currentRow];

					numMismatches++;

					previousRow = currentRow;
					previousColumn = currentColumn;

					currentRow--;
				}
				break;

			case Directions_STOP:
				keepProcessing = false;
				//mReversedAnchor[gappedAnchorLen+1]='\0';
				//mReversedQuery [gappedQueryLen+1]='\0';
				break;

			case Directions_UP:

				mReversedAnchor[gappedAnchorLen++] = s1[currentColumn];
				mReversedQuery[gappedQueryLen++]   = s2[currentRow];

				if(s1[currentColumn] != s2[currentRow]) numMismatches++;
				previousRow = currentRow;
				previousColumn = currentColumn;

				currentRow--;
				currentColumn--;
				break;

			case Directions_LEFT:
				nHorizontalGap =  mPointers[currentPosition].mSizeOfHorizontalGaps;
				for(unsigned int i = 0; i < nHorizontalGap; i++){

					mReversedAnchor[gappedAnchorLen++] = s1[currentColumn];
					mReversedQuery[gappedQueryLen++]   = GAP;

					numMismatches++;

					previousRow = currentRow;
					previousColumn = currentColumn;


					currentColumn--;
				}
				break;
		}
		currentPosition = ((currentRow + rowOffset) * (mBandwidth + 2)) + (columnOffset - currentRow + currentColumn);
	}

	// correct the reference and query sequence order
	mReversedAnchor[gappedAnchorLen] = 0;
	mReversedQuery [gappedQueryLen] = 0;
	reverse(mReversedAnchor, mReversedAnchor + gappedAnchorLen);
	reverse(mReversedQuery,  mReversedQuery  + gappedQueryLen);

	//alignment.Reference = mReversedAnchor;
	//alignment.Query     = mReversedQuery;

	// assign the alignment endpoints
	//alignment.ReferenceBegin = previousColumn;
	//alignment.ReferenceEnd   = bestColumn;
	referenceAl  = previousColumn;
	/*  
	if(alignment.IsReverseComplement){
		alignment.QueryBegin = s2.length() - bestRow - 1; 
		alignment.QueryEnd   = s2.length() - previousRow - 1;
	} else {
		alignment.QueryBegin = previousRow; 
		alignment.QueryEnd   = bestRow;
	}
	*/
	
	//alignment.QueryLength	= alignment.QueryEnd - alignment.QueryBegin + 1;
	//alignment.NumMismatches = numMismatches;

	const unsigned int alLength = strlen(mReversedAnchor);
	unsigned int m = 0, d = 0, i = 0;
	bool dashRegion = false;
	ostringstream oCigar;

	if ( previousRow != 0 )
		oCigar << previousRow << 'S';

	for ( unsigned int j = 0; j < alLength; j++ ) {
		// m
		if ( ( mReversedAnchor[j] != GAP ) && ( mReversedQuery[j] != GAP ) ) {
			if ( dashRegion ) {
				if ( d != 0 ) oCigar << d << 'D';
				else          oCigar << i << 'I';
			}
			dashRegion = false;
			m++;
			d = 0;
			i = 0;
		}
		// I or D
		else {
			if ( !dashRegion )
				oCigar << m << 'M';
			dashRegion = true;
			m = 0;
			if ( mReversedAnchor[j] == GAP ) {
				if ( d != 0 ) oCigar << d << 'D';
				i++;
				d = 0;
			}
			else {
				if ( i != 0 ) oCigar << i << 'I';
				d++;
				i = 0;
			}
		}
	}
	
	if      ( m != 0 ) oCigar << m << 'M';
	else if ( d != 0 ) oCigar << d << 'D';
	else if ( i != 0 ) oCigar << i << 'I';

	if ( ( bestRow + 1 ) != s2.length() )
		oCigar << s2.length() - bestRow - 1 << 'S';

	cigarAl = oCigar.str();
	

	// correct the homopolymer gap order
	CorrectHomopolymerGapOrder(alLength, numMismatches);

}
