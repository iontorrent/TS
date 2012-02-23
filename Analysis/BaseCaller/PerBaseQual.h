/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef PERBASEQUAL_H
#define PERBASEQUAL_H

#include <math.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <stdint.h>
#include <cstdio>
#include <vector>
#include <limits.h>

#include "RawWells.h"
#include "ChipIdDecoder.h"

const int nPredictors = 6;

class PerBasePredictors
{

    public:
        bool    InitWell( ChipIdEnum _phredVersion, int numBasesCalled, float* flowValues, float* _cafieResidual, int numFlows, int* flowIndex );
        float GetPredictor1( int base ) const;
        float GetPredictor2( int base ) const;
        float GetPredictor3( void ) const;
        float GetPredictor4( int base ) const;
        float GetPredictor5( int base ) const;
        float GetPredictor6( int base ) const;
        float GetPredictor( int pred, int base ) const;

        int NumBases( void )const  {
            return numba;
        }

        int NumFlows( void )const  {
            return numFlo;
        }
        
        void setFlowOrder( const std::string& flowOrder ) { sameNucOffset = calculateSameNucOffset( flowOrder ); }
        
        int MapFlowToBase( int flow );

        void setWellName( const std::string& _pixelName ) {
            pixelName = _pixelName;
        }

        PerBasePredictors();
        ~PerBasePredictors();

        std::vector<float> penaltyMismatch;
        std::vector<float> penaltyResidual;
        bool TreephaserUsed;

    private:
        void    CalculateCutoff( float cutoff0, float cutoff1 );
        int homcount( int e );
        
        std::vector<int> sameNucOffset; //each element is the number of flows since the last flow of the same nuc
        std::vector<int> calculateSameNucOffset( std::string flowOrder );
        
        std::string pixelName;
        int *flowIdx;
        std::vector< int > cfarray;
        int number;
        std::vector< int > homopols;
        float *flowVal;
        int numFlo;
        int *baseflow;
        float *cafieResidual;
        int numba;
        float sum_one;
        float sum_zero;
        int one_counter;
        int zero_counter;
        float mean_zero;
        float mean_one;
        float stdev_zero;
        float stdev_one;
        float sumstd_zero;
        float sumstd_one;
        float cutOff1;
        float cutOff0;
        int maxIter;
        float noiseValue, snr;
        ChipIdEnum phredVersion;
};


class PerBaseQual
{

    private:
        std::stringstream predictor_save;
        std::string wellName, phredFileName;
        bool isInitialized;
        std::vector< std::vector< float > > phredTableData;
        std::vector< float > phredTableMaxValues;

        ChipIdEnum phredVersion;
        PerBasePredictors pbq;
        int MinQScore;
        int maxFlowLimit; //after this flow we return minimum quality score.
        int CalculatePerBaseScore( int baseIndex );
        
    public:
        PerBaseQual( void ) : isInitialized( false ), phredVersion( ChipIdUnknown ), MinQScore( 5 ), maxFlowLimit(INT_MAX) {
            phredTableData.resize( nPredictors + 1 );
            phredTableMaxValues.resize( nPredictors );
        };

        bool Init( ChipIdEnum _phredVersion, const std::string& flowOrder = std::string(""), const std::string& _phredTableFile = std::string( "" ) );

        int GenerateQualityPerBaseTreephaser( std::vector<float> &_penaltyResidual, std::vector<float> &_penaltyMismatch,  weight_vec_t &correctedFlowValue, weight_vec_t &cafieResidual, std::vector< uint8_t > &baseFlowIndex, int _maxFlowLimit=INT_MAX );
        int GenerateQualityPerBase( weight_vec_t &correctedFlowValue, weight_vec_t &cafieResidual, std::vector< uint8_t > &baseFlowIndex, int _maxFlowLimit = INT_MAX );
        int GenerateQualityPerBase( int numBasesCalled, float* flowValues, float* cafieResidual, int numFlows, int* flowIndex, int _maxFlowLimit = INT_MAX );
        int GetQuality( int baseIndex );
        uint8_t *GetQualities( void );
        void GetQualities(std::vector<uint8_t> &qualityScores);
        void setWellName( const std::string& _wellName )  {
            wellName = _wellName;
            pbq.setWellName( wellName );
        };
};

#endif // PERBASEQUAL_H
