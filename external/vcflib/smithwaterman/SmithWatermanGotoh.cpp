#include "SmithWatermanGotoh.h"

const float CSmithWatermanGotoh::FLOAT_NEGATIVE_INFINITY = (float)-1e+30;

const char CSmithWatermanGotoh::Directions_STOP     = 0;
const char CSmithWatermanGotoh::Directions_LEFT     = 1;
const char CSmithWatermanGotoh::Directions_DIAGONAL = 2;
const char CSmithWatermanGotoh::Directions_UP       = 3;

const int CSmithWatermanGotoh::repeat_size_max      = 12;

CSmithWatermanGotoh::CSmithWatermanGotoh(float matchScore, float mismatchScore, float gapOpenPenalty, float gapExtendPenalty) 
    : mCurrentMatrixSize(0)
    , mCurrentAnchorSize(0)
    , mCurrentQuerySize(0)
    , mCurrentAQSumSize(0)
    , mMatchScore(matchScore)
    , mMismatchScore(mismatchScore)
    , mGapOpenPenalty(gapOpenPenalty)
    , mGapExtendPenalty(gapExtendPenalty)
    , mPointers(NULL)
    , mSizesOfVerticalGaps(NULL)
    , mSizesOfHorizontalGaps(NULL)
    , mQueryGapScores(NULL)
    , mBestScores(NULL)
    , mReversedAnchor(NULL)
    , mReversedQuery(NULL)
    , mUseHomoPolymerGapOpenPenalty(false)
    , mUseEntropyGapOpenPenalty(false)
    , mUseRepeatGapExtensionPenalty(false)
{
    CreateScoringMatrix();
}

CSmithWatermanGotoh::~CSmithWatermanGotoh(void) {
    if(mPointers)              delete [] mPointers;
    if(mSizesOfVerticalGaps)   delete [] mSizesOfVerticalGaps;
    if(mSizesOfHorizontalGaps) delete [] mSizesOfHorizontalGaps;
    if(mQueryGapScores)        delete [] mQueryGapScores;
    if(mBestScores)            delete [] mBestScores;
    if(mReversedAnchor)        delete [] mReversedAnchor;
    if(mReversedQuery)         delete [] mReversedQuery;
}

// aligns the query sequence to the reference using the Smith Waterman Gotoh algorithm
void CSmithWatermanGotoh::Align(unsigned int& referenceAl, string& cigarAl, const string& s1, const string& s2) {

    if((s1.length() == 0) || (s2.length() == 0)) {
	cout << "ERROR: Found a read with a zero length." << endl;
	exit(1);
    }

    unsigned int referenceLen      = s1.length() + 1;
    unsigned int queryLen          = s2.length() + 1;
    unsigned int sequenceSumLength = s1.length() + s2.length();

    // reinitialize our matrices

    if((referenceLen * queryLen) > mCurrentMatrixSize) {

	// calculate the new matrix size
	mCurrentMatrixSize = referenceLen * queryLen;

	// delete the old arrays
	if(mPointers)              delete [] mPointers;
	if(mSizesOfVerticalGaps)   delete [] mSizesOfVerticalGaps;
	if(mSizesOfHorizontalGaps) delete [] mSizesOfHorizontalGaps;

	try {

	    // initialize the arrays
	    mPointers              = new char[mCurrentMatrixSize];
	    mSizesOfVerticalGaps   = new short[mCurrentMatrixSize];
	    mSizesOfHorizontalGaps = new short[mCurrentMatrixSize];

	} catch(bad_alloc) {
	    cout << "ERROR: Unable to allocate enough memory for the Smith-Waterman algorithm." << endl;
	    exit(1);
	}
    }

    // initialize the traceback matrix to STOP
    memset((char*)mPointers, 0, SIZEOF_CHAR * queryLen);
    for(unsigned int i = 1; i < referenceLen; i++) mPointers[i * queryLen] = 0;

    // initialize the gap matrices to 1
    uninitialized_fill(mSizesOfVerticalGaps, mSizesOfVerticalGaps + mCurrentMatrixSize, 1);
    uninitialized_fill(mSizesOfHorizontalGaps, mSizesOfHorizontalGaps + mCurrentMatrixSize, 1);


    // initialize our repeat counts if they are needed
    vector<map<string, int> > referenceRepeats;
    vector<map<string, int> > queryRepeats;
    //int queryBeginRepeatBases = 0;
    //int queryEndRepeatBases = 0;
    if (mUseRepeatGapExtensionPenalty) {
	for (unsigned int i = 0; i < queryLen; ++i)
	    queryRepeats.push_back(repeatCounts(i, s2, repeat_size_max));
	for (unsigned int i = 0; i < referenceLen; ++i)
	    referenceRepeats.push_back(repeatCounts(i, s1, repeat_size_max));

	// keep only the biggest repeat
	vector<map<string, int> >::iterator q = queryRepeats.begin();
	for (; q != queryRepeats.end(); ++q) {
	    map<string, int>::iterator biggest = q->begin();
	    map<string, int>::iterator z = q->begin();
	    for (; z != q->end(); ++z)
		if (z->first.size() > biggest->first.size()) biggest = z;
	    z = q->begin();
	    while (z != q->end()) {
		if (z != biggest)
		    q->erase(z++);
		else ++z;
	    }
	}

	q = referenceRepeats.begin();
	for (; q != referenceRepeats.end(); ++q) {
	    map<string, int>::iterator biggest = q->begin();
	    map<string, int>::iterator z = q->begin();
	    for (; z != q->end(); ++z)
		if (z->first.size() > biggest->first.size()) biggest = z;
	    z = q->begin();
	    while (z != q->end()) {
		if (z != biggest)
		    q->erase(z++);
		else ++z;
	    }
	}

	// remove repeat information from ends of queries
	// this results in the addition of spurious flanking deletions in repeats
	map<string, int>& qrend = queryRepeats.at(queryRepeats.size() - 2);
	if (!qrend.empty()) {
	    int queryEndRepeatBases = qrend.begin()->first.size() * qrend.begin()->second;
	    for (int i = 0; i < queryEndRepeatBases; ++i)
		queryRepeats.at(queryRepeats.size() - 2 - i).clear();
	}

	map<string, int>& qrbegin = queryRepeats.front();
	if (!qrbegin.empty()) {
	    int queryBeginRepeatBases = qrbegin.begin()->first.size() * qrbegin.begin()->second;
	    for (int i = 0; i < queryBeginRepeatBases; ++i)
		queryRepeats.at(i).clear();
	}

    }

    //int entropyWindowSize = 8;
    vector<float> referenceEntropies;
    vector<float> queryEntropies;
    /*if (mUseEntropyGapOpenPenalty) {
	for (unsigned int i = 0; i < queryLen; ++i)
	    queryEntropies.push_back(
		shannon_H((char*) &s2[max(0, min((int) i - entropyWindowSize / 2, (int) queryLen - entropyWindowSize - 1))],
			  entropyWindowSize));
	for (unsigned int i = 0; i < referenceLen; ++i)
	    referenceEntropies.push_back(
		shannon_H((char*) &s1[max(0, min((int) i - entropyWindowSize / 2, (int) referenceLen - entropyWindowSize - 1))],
			  entropyWindowSize));
    }
    */
    // normalize entropies
    /*
    float qsum = 0;
    float qnorm = 0;
    float qmax = 0;
    for (vector<float>::iterator q = queryEntropies.begin(); q != queryEntropies.end(); ++q) {
	qsum += *q;
	if (*q > qmax) qmax = *q;
    }
    qnorm = qsum / queryEntropies.size();
    for (vector<float>::iterator q = queryEntropies.begin(); q != queryEntropies.end(); ++q)
	*q = *q / qsum + qmax;

    float rsum = 0;
    float rnorm = 0;
    float rmax = 0;
    for (vector<float>::iterator r = referenceEntropies.begin(); r != referenceEntropies.end(); ++r) {
	rsum += *r;
	if (*r > rmax) rmax = *r;
    }
    rnorm = rsum / referenceEntropies.size();
    for (vector<float>::iterator r = referenceEntropies.begin(); r != referenceEntropies.end(); ++r)
	*r = *r / rsum + rmax;
    */

    //
    // construct
    //

    // reinitialize our query-dependent arrays
    if(s2.length() > mCurrentQuerySize) {

	// calculate the new query array size
	mCurrentQuerySize = s2.length();

	// delete the old arrays
	if(mQueryGapScores) delete [] mQueryGapScores;
	if(mBestScores)     delete [] mBestScores;

	// initialize the arrays
	try {

	    mQueryGapScores = new float[mCurrentQuerySize + 1];
	    mBestScores     = new float[mCurrentQuerySize + 1];

	} catch(bad_alloc) {
	    cout << "ERROR: Unable to allocate enough memory for the Smith-Waterman algorithm." << endl;
	    exit(1);
	}
    }

    // reinitialize our reference+query-dependent arrays
    if(sequenceSumLength > mCurrentAQSumSize) {

	// calculate the new reference array size
	mCurrentAQSumSize = sequenceSumLength;

	// delete the old arrays
	if(mReversedAnchor) delete [] mReversedAnchor;
	if(mReversedQuery)  delete [] mReversedQuery;

	// initialize the arrays
	try {

	    mReversedAnchor = new char[mCurrentAQSumSize + 1];	// reversed sequence #1
	    mReversedQuery  = new char[mCurrentAQSumSize + 1];	// reversed sequence #2

	} catch(bad_alloc) {
	    cout << "ERROR: Unable to allocate enough memory for the Smith-Waterman algorithm." << endl;
	    exit(1);
	}
    }

    // initialize the gap score and score vectors
    uninitialized_fill(mQueryGapScores, mQueryGapScores + queryLen, FLOAT_NEGATIVE_INFINITY);
    memset((char*)mBestScores, 0, SIZEOF_FLOAT * queryLen);

    float similarityScore, totalSimilarityScore, bestScoreDiagonal;
    float queryGapExtendScore, queryGapOpenScore;
    float referenceGapExtendScore, referenceGapOpenScore, currentAnchorGapScore;

    unsigned int BestColumn = 0;
    unsigned int BestRow    = 0;
    float BestScore         = FLOAT_NEGATIVE_INFINITY;

    for(unsigned int i = 1, k = queryLen; i < referenceLen; i++, k += queryLen) {

	currentAnchorGapScore = FLOAT_NEGATIVE_INFINITY;
	bestScoreDiagonal = mBestScores[0];

	for(unsigned int j = 1, l = k + 1; j < queryLen; j++, l++) {

	    // calculate our similarity score
	    similarityScore = mScoringMatrix[s1[i - 1] - 'A'][s2[j - 1] - 'A'];

	    // fill the matrices
	    totalSimilarityScore = bestScoreDiagonal + similarityScore;
	    
	    //cerr << "i: " << i << ", j: " << j << ", totalSimilarityScore: " << totalSimilarityScore << endl;

	    queryGapExtendScore = mQueryGapScores[j] - mGapExtendPenalty;
	    queryGapOpenScore   = mBestScores[j] - mGapOpenPenalty;
	    
	    // compute the homo-polymer gap score if enabled
	    if(mUseHomoPolymerGapOpenPenalty)
		if((j > 1) && (s2[j - 1] == s2[j - 2]))
		    queryGapOpenScore = mBestScores[j] - mHomoPolymerGapOpenPenalty;
	    
	    // compute the entropy gap score if enabled
	    if (mUseEntropyGapOpenPenalty) {
		queryGapOpenScore = 
		    mBestScores[j] - mGapOpenPenalty 
		    * max(queryEntropies.at(j), referenceEntropies.at(i))
		    * mEntropyGapOpenPenalty;
	    }

	    int gaplen = mSizesOfVerticalGaps[l - queryLen] + 1;

	    if (mUseRepeatGapExtensionPenalty) {
		map<string, int>& repeats = queryRepeats[j];
		// does the sequence which would be inserted or deleted in this gap match the repeat structure which it is embedded in?
		if (!repeats.empty()) {

		    const pair<string, int>& repeat = *repeats.begin();
		    int repeatsize = repeat.first.size();
		    if (gaplen != repeatsize && gaplen % repeatsize != 0) {
			gaplen = gaplen / repeatsize + repeatsize;
		    }

		    if ((repeat.first.size() * repeat.second) > 3 && gaplen + i < s1.length()) {
			string gapseq = string(&s1[i], gaplen);
			if (gapseq == repeat.first || isRepeatUnit(gapseq, repeat.first)) {
			    queryGapExtendScore = mQueryGapScores[j]
				+ mRepeatGapExtensionPenalty / (float) gaplen;
				//    mMaxRepeatGapExtensionPenalty)
			} else {
			    queryGapExtendScore = mQueryGapScores[j] - mGapExtendPenalty;
			}
		    }
		} else {
		    queryGapExtendScore = mQueryGapScores[j] - mGapExtendPenalty;
		}
	    }
		  
	    if(queryGapExtendScore > queryGapOpenScore) {
		mQueryGapScores[j] = queryGapExtendScore;
		mSizesOfVerticalGaps[l] = gaplen;
	    } else mQueryGapScores[j] = queryGapOpenScore;
	    
	    referenceGapExtendScore = currentAnchorGapScore - mGapExtendPenalty;
	    referenceGapOpenScore   = mBestScores[j - 1] - mGapOpenPenalty;
		  
	    // compute the homo-polymer gap score if enabled
	    if(mUseHomoPolymerGapOpenPenalty)
		if((i > 1) && (s1[i - 1] == s1[i - 2]))
		    referenceGapOpenScore = mBestScores[j - 1] - mHomoPolymerGapOpenPenalty;
		  
	    // compute the entropy gap score if enabled
	    if (mUseEntropyGapOpenPenalty) {
		referenceGapOpenScore = 
		    mBestScores[j - 1] - mGapOpenPenalty 
		    * max(queryEntropies.at(j), referenceEntropies.at(i))
		    * mEntropyGapOpenPenalty;
	    }

	    gaplen = mSizesOfHorizontalGaps[l - 1] + 1;

	    if (mUseRepeatGapExtensionPenalty) {
		map<string, int>& repeats = referenceRepeats[i];
		// does the sequence which would be inserted or deleted in this gap match the repeat structure which it is embedded in?
		if (!repeats.empty()) {

		    const pair<string, int>& repeat = *repeats.begin();
		    int repeatsize = repeat.first.size();
		    if (gaplen != repeatsize && gaplen % repeatsize != 0) {
			gaplen = gaplen / repeatsize + repeatsize;
		    }

		    if ((repeat.first.size() * repeat.second) > 3 && gaplen + j < s2.length()) {
			string gapseq = string(&s2[j], gaplen);
			if (gapseq == repeat.first || isRepeatUnit(gapseq, repeat.first)) {
			    referenceGapExtendScore = currentAnchorGapScore
				+ mRepeatGapExtensionPenalty / (float) gaplen;
				//mMaxRepeatGapExtensionPenalty)
			} else {
			    referenceGapExtendScore = currentAnchorGapScore - mGapExtendPenalty;
			}
		    }
		} else {
		    referenceGapExtendScore = currentAnchorGapScore - mGapExtendPenalty;
		}
	    }

	    if(referenceGapExtendScore > referenceGapOpenScore) {
		currentAnchorGapScore = referenceGapExtendScore;
		mSizesOfHorizontalGaps[l] = gaplen;
	    } else currentAnchorGapScore = referenceGapOpenScore;
		  
	    bestScoreDiagonal = mBestScores[j];
	    mBestScores[j] = MaxFloats(totalSimilarityScore, mQueryGapScores[j], currentAnchorGapScore);
		  
		  
	    // determine the traceback direction
	    // diagonal (445364713) > stop (238960195) > up (214378647) > left (166504495)
	    if(mBestScores[j] == 0)                         mPointers[l] = Directions_STOP;
	    else if(mBestScores[j] == totalSimilarityScore) mPointers[l] = Directions_DIAGONAL;
	    else if(mBestScores[j] == mQueryGapScores[j])   mPointers[l] = Directions_UP;
	    else                                            mPointers[l] = Directions_LEFT;
		  
	    // set the traceback start at the current cell i, j and score
	    if(mBestScores[j] > BestScore) {
		BestRow    = i;
		BestColumn = j;
		BestScore  = mBestScores[j];
	    }
	}
    }

    //
    // traceback
    //

    // aligned sequences
    int gappedAnchorLen  = 0;   // length of sequence #1 after alignment
    int gappedQueryLen   = 0;   // length of sequence #2 after alignment
    int numMismatches    = 0;   // the mismatched nucleotide count

    char c1, c2;

    int ci = BestRow;
    int cj = BestColumn;
    int ck = ci * queryLen;

    // traceback flag
    bool keepProcessing = true;

    while(keepProcessing) {
	//cerr << ci << " " << cj << " " << ck << "  ... " << gappedAnchorLen << " " << gappedQueryLen <<  endl;

	// diagonal (445364713) > stop (238960195) > up (214378647) > left (166504495)
	switch(mPointers[ck + cj]) {

	case Directions_DIAGONAL:
	    c1 = s1[--ci];
	    c2 = s2[--cj];
	    ck -= queryLen;

	    mReversedAnchor[gappedAnchorLen++] = c1;
	    mReversedQuery[gappedQueryLen++]   = c2;

	    // increment our mismatch counter
	    if(mScoringMatrix[c1 - 'A'][c2 - 'A'] == mMismatchScore) numMismatches++;	
	    break;

	case Directions_STOP:
	    keepProcessing = false;
	    break;

	case Directions_UP:
	    for(unsigned int l = 0, len = mSizesOfVerticalGaps[ck + cj]; l < len; l++) {
		if (ci <= 0) {
		    keepProcessing = false;
		    break;
		}
		mReversedAnchor[gappedAnchorLen++] = s1[--ci];
		mReversedQuery[gappedQueryLen++]   = GAP;
		ck -= queryLen;
		numMismatches++;
	    }
	    break;

	case Directions_LEFT:
	    for(unsigned int l = 0, len = mSizesOfHorizontalGaps[ck + cj]; l < len; l++) {
		if (cj <= 0) {
		    keepProcessing = false;
		    break;
		}
		mReversedAnchor[gappedAnchorLen++] = GAP;
		mReversedQuery[gappedQueryLen++]   = s2[--cj];
		numMismatches++;
	    }
	    break;
	}
    }

    // define the reference and query sequences
    mReversedAnchor[gappedAnchorLen] = 0;
    mReversedQuery[gappedQueryLen]   = 0;

    // catch sequences with different lengths
    if(gappedAnchorLen != gappedQueryLen) {
	cout << "ERROR: The aligned sequences have different lengths after Smith-Waterman-Gotoh algorithm." << endl;
	exit(1);
    }

    // reverse the strings and assign them to our alignment structure
    reverse(mReversedAnchor, mReversedAnchor + gappedAnchorLen);
    reverse(mReversedQuery,  mReversedQuery  + gappedQueryLen);

    //alignment.Reference = mReversedAnchor;
    //alignment.Query     = mReversedQuery;

    // set the reference endpoints
    //alignment.ReferenceBegin = ci;
    //alignment.ReferenceEnd   = BestRow - 1;
    referenceAl = ci;

    // set the query endpoints
    /*  
	if(alignment.IsReverseComplement) {
	alignment.QueryBegin = s2Length - BestColumn;
	alignment.QueryEnd   = s2Length - cj - 1;
	// alignment.QueryLength= alignment.QueryBegin - alignment.QueryEnd + 1;
	} else {
	alignment.QueryBegin = cj;
	alignment.QueryEnd   = BestColumn - 1;
	// alignment.QueryLength= alignment.QueryEnd - alignment.QueryBegin + 1;
	}
    */

    // set the query length and number of mismatches
    //alignment.QueryLength = alignment.QueryEnd - alignment.QueryBegin + 1;
    //alignment.NumMismatches  = numMismatches;

    unsigned int alLength = strlen(mReversedAnchor);
    unsigned int m = 0, d = 0, i = 0;
    bool dashRegion = false;
    ostringstream oCigar (ostringstream::out);
    int insertedBases = 0;

    if ( cj != 0 ) {
	if ( cj > 0 ) {
	    oCigar << cj << 'S';
	} else { // how do we get negative cj's?
	    referenceAl -= cj;
	    alLength += cj;
	}
    }
	
    for ( unsigned int j = 0; j < alLength; j++ ) {
	// m
	if ( ( mReversedAnchor[j] != GAP ) && ( mReversedQuery[j] != GAP ) ) {
	    if ( dashRegion ) {
		if ( d != 0 ) oCigar << d << 'D';
		else          { oCigar << i << 'I'; insertedBases += i; }
	    }
	    dashRegion = false;
	    m++;
	    d = 0;
	    i = 0;
	}
	else {
	    if ( !dashRegion && m )
		oCigar << m << 'M';
	    dashRegion = true;
	    m = 0;
	    if ( mReversedAnchor[j] == GAP ) {
		if ( d != 0 ) oCigar << d << 'D';
		i++;
		d = 0;
	    }
	    else {
		if ( i != 0) { oCigar << i << 'I'; insertedBases += i; }
		d++;
		i = 0;
	    }
	}
    }
    if      ( m != 0 ) oCigar << m << 'M';
    else if ( d != 0 ) oCigar << d << 'D';
    else if ( i != 0 ) oCigar << i << 'I';

    if ( BestColumn != s2.length() )
	oCigar << s2.length() - BestColumn << 'S';

    cigarAl = oCigar.str();

    // fix the gap order
    CorrectHomopolymerGapOrder(alLength, numMismatches);

    if (mUseEntropyGapOpenPenalty || mUseRepeatGapExtensionPenalty) {
	int offset = 0;
	string oldCigar;
	try {
	    oldCigar = cigarAl;
	    stablyLeftAlign(s2, cigarAl, s1.substr(referenceAl, alLength - insertedBases), offset);
	} catch (...) {
	    cerr << "an exception occurred when left-aligning " << s1 << " " << s2 << endl;
	    cigarAl = oldCigar; // undo the failed left-realignment attempt
	    offset = 0;
	}
	referenceAl += offset;
    }

}

// creates a simple scoring matrix to align the nucleotides and the ambiguity code N
void CSmithWatermanGotoh::CreateScoringMatrix(void) {

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
void CSmithWatermanGotoh::EnableHomoPolymerGapPenalty(float hpGapOpenPenalty) {
    mUseHomoPolymerGapOpenPenalty = true;
    mHomoPolymerGapOpenPenalty    = hpGapOpenPenalty;
}

// enables entropy-based gap open penalty
void CSmithWatermanGotoh::EnableEntropyGapPenalty(float enGapOpenPenalty) {
    mUseEntropyGapOpenPenalty = true;
    mEntropyGapOpenPenalty    = enGapOpenPenalty;
}

// enables repeat-aware gap extension penalty
void CSmithWatermanGotoh::EnableRepeatGapExtensionPenalty(float rGapExtensionPenalty, float rMaxGapRepeatExtensionPenaltyFactor) {
    mUseRepeatGapExtensionPenalty = true;
    mRepeatGapExtensionPenalty    = rGapExtensionPenalty;
    mMaxRepeatGapExtensionPenalty = rMaxGapRepeatExtensionPenaltyFactor * rGapExtensionPenalty;
}

// corrects the homopolymer gap order for forward alignments
void CSmithWatermanGotoh::CorrectHomopolymerGapOrder(const unsigned int numBases, const unsigned int numMismatches) {

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
