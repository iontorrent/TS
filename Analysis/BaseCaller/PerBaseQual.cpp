/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */

#include "PerBaseQual.h"
#include <algorithm>
#include <math.h>
#include <limits.h>
#include <iomanip>
#include <stdio.h>

#include "IonErr.h"

using namespace std;
#define BASE_SIZE 30000
#define DEFAULT_PHRED_TABLE_NAME "phredTable.txt"

//#define DUMP_PREDICTORS

#ifdef DUMP_PREDICTORS
class PredictorSaver{
    ofstream predictor_dump;
    pthread_spinlock_t spinlock;
    bool isInitialized;
public:
    PredictorSaver() : isInitialized(false){
        pthread_spin_init(&spinlock, 0);
    }

    void Init( void ){
        pthread_spin_lock(&spinlock);
        if( !isInitialized ){
            predictor_dump.open("Predictors.txt");
            isInitialized = true;
        }
        pthread_spin_unlock(&spinlock);
    }
    
    void Save( const std::stringstream & str ){
        pthread_spin_lock(&spinlock);
        predictor_dump << str.str();
        pthread_spin_unlock(&spinlock);
    }
    
    ~PredictorSaver(){
        pthread_spin_destroy(&spinlock);
    }
};

PredictorSaver predictorSaver;
#endif 

PerBasePredictors::PerBasePredictors( )
{
    baseflow = ( int* ) calloc( 1, sizeof( int ) * BASE_SIZE );
    flowVal = ( float* ) calloc( 1, sizeof( float ) * BASE_SIZE );
    cafieResidual = ( float* ) calloc( 1, sizeof( float ) * BASE_SIZE );
    flowIdx = ( int* ) calloc( 1, sizeof( int ) * BASE_SIZE );
    TreephaserUsed = false;
}

PerBasePredictors::~PerBasePredictors()
{
    if ( baseflow )
        free( baseflow );

    if ( flowVal )
        free( flowVal );

    if ( cafieResidual )
        free( cafieResidual );

    if ( flowIdx )
        free( flowIdx );

}

//initialize quality calculator for the current well
bool PerBasePredictors::InitWell( ChipIdEnum _phredVersion, int numBasesCalled, float* flowValues, float* _cafieResidual, int numFlows, int* flowIndex )
{
    phredVersion = _phredVersion;
    numba = numBasesCalled;
    numFlo = numFlows;

    /* Go from float to 100-based integer accuracy and back to float (e.g. 1.16945 -> 1.17) */

    for ( int flo = 0; flo < numFlo; ++flo ) {
        cafieResidual[flo] = _cafieResidual[flo];
        flowVal[flo] = ( rint( 100 * flowValues[flo] ) ) / 100.0;

        if ( flowVal[flo] < 0 ) {
            flowVal[flo] = 0.0;
        }
    }

    /* convert flow intensity to base-intensity */

    for ( int a = 0; a < NumBases(); a++ ) {
        flowIdx[a] = flowIndex[a];
    }

    int sum = -1;

    for ( int b = 0; b < NumBases();b++ ) {
        sum += flowIdx[b];
        baseflow[b] = sum;
    }


    /* create homopolymer, cafie arrays; use the flow index to identify how
     *    many base per flow were called and which homopolymer value to assign
     *    to how many bases */
    homopols.clear();
    homopols.resize(NumBases(), 1);
    
    // homopolymer count
    number = 1;

    int u = 0;

    // process read
    while ( u < NumBases() ) {
        number = 1;
        // recursive; homopolymer length
        number = homcount( u );
        // HP 333
        //for(int g=0; g<number; g++){
        // homopols[u+g]=number;
        //}

        // HP 1124
        //for(int g=0; g<number/2; g++){
        //  homopols[u+g]=1;
        //}
        //int rest = number - number/2;
        //for(int g=(int)(number/2); g<number; g++){
        //  homopols[u+g]=number/2;
        //}
        //homopols[u+number-1]=number;

        // HP 1114

        for ( int g = 0; g < number - 1; g++ ) {
            homopols.at(u+g) = 1;
        }

        homopols.at(u+number-1) = number;

        // go to next position in read after the homopolymer
        u = u + number;
    }

    /* 0-mer and 1-mer overlap */
    /* define 0-mer and 1-mer interval */
    cutOff0 = 0.5;

    cutOff1 = 1.5;

    maxIter = 2;

    /* adjust cutoffs once */
    for ( int i = 0; i < maxIter; i++ ) {
        CalculateCutoff( cutOff0, cutOff1 );
    }

    noiseValue = ( mean_one - mean_zero - stdev_one - stdev_zero ) / ( mean_one );

    return true;
}

/*Map flow to the next base*/
int PerBasePredictors::MapFlowToBase( int flow )
{
    for ( int i = 0; i < NumBases(); ++i ) {
        if ( flow <= baseflow[ i ] )
            return i;
    }

    return NumBases();
}

/* to calculate homopolymer count per base; use flow index to identify how many bases were called in each flow */
int PerBasePredictors::homcount( int s )
{
    if ( ((s+1) < NumBases()) && (baseflow[s] == baseflow[s+1]) ) {
        s++;
        // homopolymer count
        number++;
        homcount( s );
    }

    return number;
}

float PerBasePredictors::GetPredictor1( int base ) const
{
   	return penaltyResidual.at(base);
}

/* Predictor 2 - Local noise/flowalign - 'noise' in the input base's measured val.  Noise is max[abs(val - round(val))] within a radius of 3 BASES; 12 bins */
float PerBasePredictors::GetPredictor2( int base ) const
{
    float locnoise = 0;

    /* protect at start/end of read */
    int val1 = base - 1;
    int val2 = base + 1;

    if ( val1 < 0 ) {
        val1 = 0;
    }

    if ( val2 >= NumBases() ) {
        val2 = NumBases() - 1;
    }

    for ( int j = val1;j <= val2 && j < BASE_SIZE;j++ ) {
        if ( fabsf( flowVal[baseflow[j]] - roundf( flowVal[baseflow[j]] ) ) > locnoise ) {
            locnoise = fabsf( flowVal[baseflow[j]] - roundf( flowVal[baseflow[j]] ) );
        }
    }

    return locnoise;
}


/* Predictor 3  - Read Noise/Overlap - mean & stdev of the 0-mers & 1-mers in the read
 * -(m_1 - m_0 - s_1 - s_0)/m_1; */
float PerBasePredictors::GetPredictor3( void ) const
{
    return -noiseValue;
}

/* recalculate Offset for Predictor6
 * NewCutOff0 = (s_0 * m_1 + s_1 * m_0)/(s_1 + s_0)
 * NewCutoff1 = 2 * m_1 - NewCutOff0  */
void PerBasePredictors::CalculateCutoff( float cutoff0, float cutoff1 )
{
    sum_one = 0.0;
    sum_zero = 0.0;
    one_counter = 0;
    zero_counter = 0;
    mean_zero = 0.0;
    mean_one = 0.0;
    stdev_zero = 0.0;
    stdev_one = 0.0;
    sumstd_zero = 0.0;
    sumstd_one = 0.0;


    int numFloPred = std::min( numFlo, 60 );

    for ( int j = 8; j < numFloPred; j++ ) {
        if (( flowVal[j] < 0 ) ) {
            flowVal[j] = 0;
            zero_counter++;
        }

        if ( flowVal[j] < cutoff0 ) {
            sum_zero += flowVal[j];
            zero_counter++;
        }

        if (( flowVal[j] >= cutoff0 ) && ( flowVal[j] < cutoff1 ) ) {
            sum_one += flowVal[j];
            one_counter++;
        }
    }

    /* if run has no 0-mers or 1-mers */

    if ( zero_counter == 0 ) {
        mean_zero = 0;
    }

    if ( zero_counter != 0 ) {
        mean_zero = sum_zero / ( float )( zero_counter );
    }

    if ( one_counter == 0 ) {
        mean_one = 1;
    }

    if ( one_counter != 0 ) {
        mean_one = sum_one / ( float )( one_counter );
    }


    /* for reads with only 0-mers or 1-mers - throw exception */

    for ( int n = 8; n < numFloPred; n++ ) {
        if ( flowVal[n] < 0 ) {
            flowVal[n] = 0;
        }

        if (( flowVal[n] >= 0 ) && ( flowVal[n] < cutoff0 ) ) {
            sumstd_zero += ( flowVal[n] - mean_zero ) * ( flowVal[n] - mean_zero );
        }

        if (( flowVal[n] >= cutoff0 ) && ( flowVal[n] < cutoff1 ) ) {
            sumstd_one += ( flowVal[n] - mean_one ) * ( flowVal[n] - mean_one );
        }
    }

    /* if run has no 0-mers or 1-mers */

    if ( zero_counter == 0 ) {
        stdev_zero = 0;
    }

    if ( zero_counter != 0 ) {
        stdev_zero = sqrt( sumstd_zero / ( float )( zero_counter ) );
    }

    if ( one_counter == 0 ) {
        stdev_one = 0;
    }

    if ( one_counter != 0 ) {
        stdev_one = sqrt( sumstd_one / ( float )( one_counter ) );
    }

    /* calculate new cutoffs */

    if (( stdev_one > 0 ) || ( stdev_zero > 0 ) ) {
        cutOff0 = ( mean_one * stdev_zero + mean_zero * stdev_one ) / ( stdev_one + stdev_zero );
        cutOff1 = 2 * mean_one - cutOff0;
    }

    else {
        cutOff0 = 0.5;
        cutOff1 = 1.5;
    }
}



/* Predictor 4 - Homopolymer count - # of consecutive bases equal to the input base, including itself; 6 bins */
float PerBasePredictors::GetPredictor4( int base ) const
{
    return homopols.at(base);
}


/* Predictor 5 - CAFIE error - the number of bases identical to the current flow in the previous cycle; 7 bins */
// For Treephaser: Penalty indicating deletion after the called base
float PerBasePredictors::GetPredictor5( int base ) const
{
   	return penaltyMismatch.at(base);
}


/* Predictor 6 - Local noise - max of 'noise' 10 BASES FORWARD around a base.  Noise is max{abs(val - round(val))}; 20 bins */
float PerBasePredictors::GetPredictor6( int base ) const
{

    float noise = 0;

    int bandWidth = 5;
    /* protect at start/end of read */
    int val1 = base - bandWidth;
    int val2 = base + bandWidth;

    if ( val1 < 0 ) {
        val1 = 0;
    }

    if ( val2 >= NumBases() ) {
        val2 = NumBases() - 1;
    }

    int nCount = 0;

    for ( int j = val1;j <= val2 && j < BASE_SIZE;j++ ) {
		noise += fabsf( flowVal[baseflow[j]] - roundf( flowVal[baseflow[j]] ) );
		nCount++;
    }

	if ( nCount > 0 )
		noise /= nCount;

    return noise;
}

float PerBasePredictors::GetPredictor( int pred, int base ) const
{
    switch ( pred ) {

        case 0:
            return GetPredictor1( base );

        case 1:
            return GetPredictor2( base );

        case 2:
            return GetPredictor3();

        case 3:
            return GetPredictor4( base );

        case 4:
            return GetPredictor5( base );

        case 5:
            return GetPredictor6( base );

        default:
            throw std::string( "Wrong predictor index" );
    }
}


std::vector<int> PerBasePredictors::calculateSameNucOffset( std::string flowOrder )
{
    
    if( flowOrder.empty() ){
        ION_WARN( "PerBaseQual: unknown flow order. Assuming TACG." );
        flowOrder = string("TACG");
    }
    
    std::vector< int > offset;
    offset.resize( flowOrder.size() );

    for( int k= flowOrder.size()-1; k >= 0; --k ){
        char nuc = flowOrder.at(k);
        
        offset[k] = 0;
        for( int n = k-1; n >= -(int)flowOrder.size(); --n ){
            if( nuc == flowOrder.at((n>=0)?n:n+flowOrder.size()) ){
                offset.at(k) = (k-n); 
                break;
            }
        }
    }
    
/*    for( int k = 0; k < (int)flowOrder.size(); ++k )
        cout << setw( 3 ) << flowOrder.at(k);
    cout << endl;
    for( int k = 0; k < (int)offset.size(); ++k )
        cout << setw( 3 ) << offset.at(k);
    cout << endl << endl;*/
    
    return offset;
}


bool PerBaseQual::Init( ChipIdEnum _phredVersion, const std::string& flowOrder, const std::string& _phredTableFile )
{
    ifstream source;
    
    pbq.setFlowOrder( flowOrder );

    phredVersion = _phredVersion;
    MinQScore = 5;

    //cerr << "PerBaseQual::Init. PhredVersion: " << phredVersion << endl;

    if( !_phredTableFile.empty() ){
        phredFileName = _phredTableFile;
    }
    else{
    	switch( phredVersion ){
			case ChipId314:
				phredFileName.assign( "_314" );
				break;
			case ChipId316:
				phredFileName.assign( "_316" );
				break;
			case ChipId318:
				phredFileName.assign( "_318" );
				break;
			default:
				cout << "phredVersion = " << phredVersion << ", use default "<< ChipId314 << endl;
				//ION_ABORT( "ERROR: unexpected phred score version requested." );
				phredFileName.assign( "_314" );
				break;
    	}
        
        char* fName = GetIonConfigFile(( string( DEFAULT_PHRED_TABLE_NAME ) + phredFileName ).c_str() );
        if( fName == NULL )
            ION_ABORT( "ERROR: Can't find phredTable file." );
        
        phredFileName.assign( fName );
        
	}
    
    source.open( phredFileName.c_str() );
    if ( !source.is_open() ) {
        char errorMsg[1024];
        sprintf( errorMsg, "ERROR: Finding file: %s\n", phredFileName.c_str() );
        ION_ABORT( errorMsg );
    }

    string line;

    while ( !source.eof() ) {
        std::getline( source, line );

        if ( line.empty() )
            break;

        if ( line[0] == '#' )
            continue;

        std::stringstream strs( line );

        float temp;

        for ( int k = 0; k < nPredictors; ++k ) {
            strs >> temp;
            phredTableData[k].push_back( temp );
        }

        strs >> temp; //skip n entries per bins

        strs >> temp;
        phredTableData[nPredictors].push_back( temp );
    }

    for ( int k = 0; k < nPredictors; ++k ) {
        phredTableMaxValues[k] = *( std::max_element( phredTableData[k].begin(), phredTableData[k].end() ) );
    }

#ifdef DUMP_PREDICTORS
    predictorSaver.Init();
#endif
    isInitialized = true;

    return isInitialized;
}

int PerBaseQual::CalculatePerBaseScore( int base )
{
    if ( !isInitialized ) {
        ION_ABORT( "ERROR: GenerateQualityPerBase called without being initialized." );
    }
    
    if ( base > pbq.NumBases() )
        return MinQScore;

    //TODO: this is a temporary fix for very long sequences that are sometimes generated by the basecaller
    if ( base > ( int )( .75*pbq.NumFlows() ) )
        return MinQScore;

    int nPhredCuts = phredTableData[0].size();

    float pred[nPredictors];

    for ( int k = 0; k < nPredictors; ++k ) {
        pred[k] = pbq.GetPredictor( k, base );
    }

#ifdef DUMP_PREDICTORS
    predictor_save << wellName << " " << base << " ";
    for ( int k = 0; k < nPredictors; ++k ) {
        predictor_save << pred[k] << " ";
    }
    predictor_save << endl;
#endif

    for (int k = 0; k < nPredictors; k++)
    	pred[k] = std::min(pred[k],phredTableMaxValues[k]);
    
    for ( int j = 0; j < nPhredCuts; ++j ) {
        bool ret = true;

//        for ( int k = 0; k < nPredictors; ++k ) {
//            ret = ret && ( pred[k] <= phredTableData[k][j] );
//        }
        for ( int k = 0; k < nPredictors; ++k ) {
        	if (pred[k] > phredTableData[k][j]) {
        		ret = false;
        		break;
        	}
        }

        if ( ret ) {
            //cerr << j << " " << static_cast<int>( phredTableData[nPredictors][j] ) << endl;
            return static_cast<int>( phredTableData[nPredictors][j] );
        }
    }

    return MinQScore; //minimal quality score
}

// Treephaser call to GenerateQualityPerBase
int PerBaseQual::GenerateQualityPerBaseTreephaser( std::vector<float>& _penaltyResidual, std::vector<float>& _penaltyMismatch,  weight_vec_t& correctedFlowValue,
		weight_vec_t& cafieResidual, std::vector< uint8_t >& baseFlowIndex, int _maxFlowLimit )
{
	pbq.penaltyResidual = _penaltyResidual;
	pbq.penaltyMismatch  = _penaltyMismatch;

	pbq.TreephaserUsed = true;
	int returnValue = GenerateQualityPerBase(correctedFlowValue, cafieResidual, baseFlowIndex, _maxFlowLimit );
	//pbq.TreephaserUsed = false;

	return returnValue;
}


// Overloaded call to GenerateQualityPerBase
int PerBaseQual::GenerateQualityPerBase( weight_vec_t& correctedFlowValue, weight_vec_t& cafieResidual, std::vector< uint8_t >& baseFlowIndex, int _maxFlowLimit )
{

    int numFlows = correctedFlowValue.size();
    float *flowValues = new float[numFlows];
    float *cafieRes = new float[numFlows];

    for ( int iFlow = 0; iFlow < numFlows; iFlow++ ) {
        flowValues[iFlow] = correctedFlowValue[iFlow];
        cafieRes[ iFlow ] = cafieResidual[ iFlow ];
    }

    int numBasesCalled = baseFlowIndex.size();

    int *flowIndex = new int[numBasesCalled];

    for ( int iBase = 0; iBase < numBasesCalled; iBase++ )
        flowIndex[iBase] = baseFlowIndex[iBase];

    int returnVal = GenerateQualityPerBase( numBasesCalled, flowValues, cafieRes, numFlows, flowIndex, _maxFlowLimit );

    delete [] flowValues;

    delete [] cafieRes;

    delete [] flowIndex;

    return( returnVal );
}

int PerBaseQual::GenerateQualityPerBase( int numBasesCalled, float* flowValues, float* cafieResidual, int numFlows, int* flowIndex, int _maxFlowLimit )
{
    maxFlowLimit = _maxFlowLimit;
#ifdef DUMP_PREDICTORS
    predictor_save.str(string(""));
#endif
    return pbq.InitWell( phredVersion, numBasesCalled, flowValues, cafieResidual, numFlows, flowIndex );
}

int PerBaseQual::GetQuality( int baseIndex )
{
    return CalculatePerBaseScore( baseIndex );
}

// MS: This function is potentially obsolete
uint8_t *PerBaseQual::GetQualities()
{
    int i;
    uint8_t *qualityScores;

    qualityScores = ( uint8_t * )calloc( pbq.NumBases(), sizeof( uint8_t ) );

    int maxBase = pbq.MapFlowToBase( maxFlowLimit );

    for ( i = 0;i < pbq.NumBases();i++ ) {
        if ( i < maxBase )
            qualityScores[i] = ( uint8_t )CalculatePerBaseScore( i );
        else
            qualityScores[i] = ( uint8_t )MinQScore;
    }

    return qualityScores;
}

void PerBaseQual::GetQualities(std::vector<uint8_t> &qualityScores)
{
    int maxBase = pbq.MapFlowToBase( maxFlowLimit );

    qualityScores.clear();

    for (int iBase = 0; iBase < pbq.NumBases(); iBase++) {
        if ( iBase < maxBase )
            qualityScores.push_back(( uint8_t )CalculatePerBaseScore(iBase));
        else
            qualityScores.push_back(( uint8_t )MinQScore);
    }

#ifdef DUMP_PREDICTORS
    predictor_save.flush();
    predictorSaver.Save( predictor_save );
    predictor_save.str(string(""));
#endif

}
