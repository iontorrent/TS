/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef EMPHASISVECTOR_H
#define EMPHASISVECTOR_H

#include <vector>
#include "BkgMagicDefines.h"
//#include "MathOptim.h"
#include "Serialization.h"
#include "TimeControl.h"
#include "TimeCompression.h"

class EmphasisClass
{
  public: // yes, bad form
    // data for applying emphasis to data points during fitting
    int                numEv;                        // number of emphasis vectors allocated
    std::vector<float> emphasis_vector_storage;      // storage for emphasis vectors
    float**            EmphasisVectorByHomopolymer;  // array of pointers to different vectors
    std::vector<float> EmphasisScale;                // scaling factor for each vector
    std::vector<int>   nonZeroEmphasisFrames; // number of non zero frame values for each emphasis vector

    float              emp[NUMEMPHASISPARAMETERS];   // parameters for emphasis vector generation
    bool point_emphasis_by_compression; // avoid emphasis artifacts due to highly compressed points

    // keep timing parameters as well
    float emphasis_width;   // parameters scaling the emphasis vector
    float emphasis_ampl;    // parameters scaling the emphasis vector

    DataWeightDefaults data_weights; // real control

    // timing parameters - warning, if time-compression changes these need to be updated
    std::vector<int>   my_frames_per_point;
    std::vector<float> my_frameNumber;
    int                npts; // how long the vector should be

    void CustomEmphasis ( float *evect, float evSel );
    void GenerateEmphasis ( int tsize, float t_center, int *frames_per_point, float *frameNumber, float amult,float width, float ampl );
    void Allocate ( int tsize );
    void Destroy();
    void DefaultValues();
    void SetDefaultValues ( float *, float, float );
    void BuildCurrentEmphasisTable ( float t_center, float amult );
    void SetupEmphasisTiming ( int _npts, int *frames_per_point, float *frameNumber );
    int  ReportUnusedPoints ( float threshold, int min_used );
    void SaveEmphasisVector();
    EmphasisClass();
    ~EmphasisClass();

    void SetUpEmphasis(TimeAndEmphasisDefaults &data_control, TimeCompression &time_c);

  private:
    void DetermineNonZeroEmphasisFrames(int hp);
    void AllocateScratch();
    bool restart;
    friend class boost::serialization::access;
    // Boost serialization support:
    template<class Archive>
    void load ( Archive& ar, const unsigned version )
    {
      // fprintf(stdout, "Serialization: load EmphasisVector... ");
      ar
      & numEv
      & emphasis_vector_storage
      // & EmphasisVectorByHomopolymer // rebuilt in CurrentEmphasis
      // & EmphasisScale               // rebuilt in CurrentEmphasis
      // CurrentEmphasis called in SetCrudeEmphasisVectors
      // by the RegionalizedData object that owns this EmphasisVector
      & emp
      & nonZeroEmphasisFrames
      & emphasis_width
      & emphasis_ampl
      & my_frames_per_point
      & my_frameNumber
      & npts
          & data_weights
      & point_emphasis_by_compression;

      if ( npts > 0 )
        AllocateScratch();
      // fprintf(stdout, "done EmphasisVector\n");
    }
    template<class Archive>
    void save ( Archive& ar, const unsigned version ) const
    {
      // fprintf(stdout, "Serialization: save EmphasisVector... ");
      // fprintf(stdout, "EmphasisVector: npts=%d\n", npts);
      ar
      & numEv
      & emphasis_vector_storage
      // & e.EmphasisVectorByHomopolymer[i];
      // & EmphasisScale
      & emp
      & nonZeroEmphasisFrames
      & emphasis_width
      & emphasis_ampl
      & my_frames_per_point
      & my_frameNumber
      & npts
          & data_weights
      & point_emphasis_by_compression;

      // fprintf(stdout, "done EmphasisVector\n");
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

int GenerateIndividualEmphasis ( float *vect, int vn, float *emp, int tsize, float t_center, const std::vector<int>& frames_per_point, const std::vector<float>& frameNumber, float amult,float width, float ampl );

#endif // EMPHASISVECTOR_H
