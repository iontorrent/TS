/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <map>
#include <assert.h>
#include <iostream>

using namespace std;

#include "Utils.h"
#include "RawWells.h"
#include "FileEquivalent.h"


SFFInfo::SFFInfo()
{
  read_header_length = 0;
  name_length = 0;
  number_of_bases = 0;
  clip_qual_left = 0;
  clip_qual_right = 0;
  clip_adapter_left = 0;
  clip_adapter_right = 0;
}

SFFInfo::SFFInfo ( const sff_t *info )
{
  read_header_length = info->rheader->rheader_length;
  name_length = info->rheader->name_length;
  number_of_bases = sff_n_bases ( info );
  clip_qual_left = sff_clip_qual_left ( info );
  clip_qual_right = sff_clip_qual_right ( info );
  clip_adapter_right = sff_clip_adapter_right ( info );
  clip_adapter_left = sff_clip_adapter_left ( info );
  name = string ( sff_name ( info ), info->rheader->name_length );
  flowgram_values.resize ( info->gheader->flow_length );
  copy ( &sff_flowgram ( info ) [0],&sff_flowgram ( info ) [0] + flowgram_values.size(), flowgram_values.begin() );
  flow_index_per_base.resize ( sff_n_bases ( info ) );
  copy ( &sff_flow_index ( info ) [0],&sff_flow_index ( info ) [0] + sff_n_bases ( info ), flow_index_per_base.begin() );

  bases = string ( sff_bases ( info ), sff_n_bases ( info ) );

  quality_scores.resize ( sff_n_bases ( info ) );
  copy ( &sff_quality ( info ) [0], &sff_quality ( info ) [0] + sff_n_bases ( info ), quality_scores.begin() );
}


vector<SffComparison> SFFInfo::CompareSFF ( const std::string &queryFile, const std::string &goldFile, double epsilon,
    int &found, int &missing, int &goldOnly )
{
  SffComparison compare;
  NumericalComparison<double> flowStats ( epsilon );
  NumericalComparison<double> flowIndexStats ( 0 );
  NumericalComparison<double> qScoreStats ( 0 );
  NumericalComparison<double> baseStats ( 0 );
  NumericalComparison<double> clipQualLeft ( 0 );
  NumericalComparison<double> clipQualRight ( 0 );
  NumericalComparison<double> clipAdaptLeft ( 0 );
  NumericalComparison<double> clipAdaptRight ( 0 );

  SFFWrapper querySff;
  querySff.OpenForRead ( queryFile.c_str() );

  SFFWrapper goldSff;
  int goldReads = 0;
  goldSff.OpenForRead ( goldFile.c_str() );

  unsigned int numFlows = querySff.GetHeader()->flow_length;
  goldReads = goldSff.GetHeader()->n_reads;
  // Some basic sanity checks to see if these files are compatible at all
  strncmp ( querySff.GetHeader()->flow->s, goldSff.GetHeader()->flow->s, numFlows ) == 0 || Abort ( "Different flow orders." );
  strncmp ( querySff.GetHeader()->key->s, goldSff.GetHeader()->key->s, querySff.GetHeader()->key_length ) == 0 || Abort ( "Different key sequence." );
  querySff.GetHeader()->key_length == goldSff.GetHeader()->key_length || Abort ( "Different key key lengths." );
  querySff.GetHeader()->version == goldSff.GetHeader()->version || Abort ( "Different sff version." );
  querySff.GetHeader()->flow_length == goldSff.GetHeader()->flow_length || Abort ( "Different number of flows." );
  querySff.GetHeader()->flowgram_format == goldSff.GetHeader()->flowgram_format || Abort ( "Different flowgram format code." );

  bool success=true;
  const sff_t *sff;
  map<string, SFFInfo> goldSffMap;
  while ( success && ( NULL != ( sff = goldSff.LoadNextEntry ( &success ) ) ) )
  {
    SFFInfo info ( sff );
    goldSffMap[info.name] = info;
  }
  goldSff.Close();

  found = 0;
  missing = 0;
  success=true;
  while ( success && ( NULL != ( sff = querySff.LoadNextEntry ( &success ) ) ) )
  {
    size_t numBases = sff_n_bases ( sff );
    map<string, SFFInfo>::iterator it = goldSffMap.find ( sff_name ( sff ) );
    compare.total++;
    if ( it == goldSffMap.end() )
    {
      missing++;
      compare.missing++;
    }
    else
    {
      clipQualLeft.AddPair ( it->second.clip_qual_left, sff_clip_qual_left ( sff ) );
      clipQualRight.AddPair ( it->second.clip_qual_right, sff_clip_qual_right ( sff ) );
      clipAdaptLeft.AddPair ( it->second.clip_adapter_left, sff_clip_adapter_left ( sff ) );
      clipAdaptRight.AddPair ( it->second.clip_adapter_right, sff_clip_adapter_right ( sff ) );
      found++;
      for ( size_t i = 0; i < numFlows; i++ )
      {
        flowStats.AddPair ( it->second.flowgram_values[i], sff_flowgram ( sff ) [i] );
      }
      for ( size_t i = 0; i < numBases; i++ )
      {
        flowIndexStats.AddPair ( it->second.flow_index_per_base[i], sff_flow_index ( sff ) [i] );
        baseStats.AddPair ( it->second.bases[i], sff_bases ( sff ) [i] );
        qScoreStats.AddPair ( it->second.quality_scores[i], sff_quality ( sff ) [i] );
      }
    }
  }
  vector<SffComparison> report;
  report.push_back ( SffComparison ( "flow-values", flowStats.GetNumSame(),
                                     flowStats.GetNumDiff(), flowStats.GetCount(),
                                     missing, flowStats.GetCorrelation() ) );
  report.push_back ( SffComparison ( "qscore", qScoreStats.GetNumSame(),
                                     qScoreStats.GetNumDiff(), qScoreStats.GetCount(),
                                     missing, qScoreStats.GetCorrelation() ) );
  report.push_back ( SffComparison ( "flow-indices", flowIndexStats.GetNumSame(),
                                     flowIndexStats.GetNumDiff(), flowIndexStats.GetCount(),
                                     missing, flowIndexStats.GetCorrelation() ) );
  report.push_back ( SffComparison ( "bases", baseStats.GetNumSame(),
                                     baseStats.GetNumDiff(), baseStats.GetCount(),
                                     missing, baseStats.GetCorrelation() ) );
  report.push_back ( SffComparison ( "clip-qual-left", clipQualLeft.GetNumSame(),
                                     clipQualLeft.GetNumDiff(), clipQualLeft.GetCount(),
                                     missing, clipQualLeft.GetCorrelation() ) );
  report.push_back ( SffComparison ( "clip-qual-right", clipQualRight.GetNumSame(),
                                     clipQualRight.GetNumDiff(), clipQualRight.GetCount(),
                                     missing, clipQualRight.GetCorrelation() ) );
  report.push_back ( SffComparison ( "clip-adapter-left", clipAdaptLeft.GetNumSame(),
                                     clipAdaptLeft.GetNumDiff(), clipAdaptLeft.GetCount(),
                                     missing, clipAdaptLeft.GetCorrelation() ) );
  report.push_back ( SffComparison ( "clip-adapter-right", clipAdaptRight.GetNumSame(),
                                     clipAdaptRight.GetNumDiff(), clipAdaptRight.GetCount(),
                                     missing, clipAdaptRight.GetCorrelation() ) );

  goldOnly = goldReads - found;
  querySff.Close();
  return report;
}

bool SFFInfo::Abort ( const std::string &msg )
{
  cerr << msg << endl;
  exit ( 1 );
  return false;
}

NumericalComparison<double> CompareRawWells ( const string &queryFile, const string &goldFile,
    float epsilon, double maxAbsVal )
{

  NumericalComparison<double> compare ( epsilon );
  string queryDir, queryWells, goldDir, goldWells;

  FillInDirName ( queryFile, queryDir, queryWells );
  FillInDirName ( goldFile, goldDir, goldWells );

  RawWells queryW ( queryDir.c_str(), queryWells.c_str() );
  RawWells goldW ( goldDir.c_str(), goldWells.c_str() );

  struct WellData goldData;
  goldData.flowValues = NULL;
  struct WellData queryData;
  queryData.flowValues = NULL;

  queryW.OpenForRead();
  goldW.OpenForRead();
  unsigned int numFlows = goldW.NumFlows();
  while ( !queryW.ReadNextRegionData ( &queryData ) )
  {
    assert ( !goldW.ReadNextRegionData ( &goldData ) );
    for ( unsigned int i = 0; i < numFlows; i++ )
    {
      if ( isfinite ( queryData.flowValues[i] ) && isfinite ( goldData.flowValues[i] ) &&
           ( fabs ( queryData.flowValues[i] ) < maxAbsVal && fabs ( goldData.flowValues[i] ) < maxAbsVal ) )
      {
        compare.AddPair ( queryData.flowValues[i], goldData.flowValues[i] );
      }
    }
  }

  queryW.Close();
  goldW.Close();
  /*const SampleStats<double> ssX = */
  compare.GetXStats();
  /*const SampleStats<double> ssY = */
  compare.GetYStats();
  return compare;
}
