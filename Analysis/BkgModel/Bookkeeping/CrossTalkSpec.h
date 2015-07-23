/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CROSSTALKSPEC_H
#define CROSSTALKSPEC_H

#include <vector>
#include <sstream>
#include "Serialization.h"
#include "json/json.h"

// this specifies the effect of nearest neighbors on the well known as "crosstalk"
// we have a global background trace for a region
// this specifies the ways the neighbors contribute to a "crosstalk" trace added to the global trace
// neighbors affect the cross-talk trace by
// a) providing new hydrogen ions at a rate determined by the properties & amplitude of the neighbor
// b) buffering the hydrogen ions in the box above the well in some means
// c) the residence time/buffering of the bulk determines how fast we contribute to this measure
// This would be easier if we had an "inferred bulk" trace already instead of an "average case empty"

class TraceCrossTalkSpecification
{
  public:

    int nei_affected;
    // coordinates of neighbor being summed
    std::vector<int> cx;
    std::vector<int> cy;

    // Earl's reformulated parameters
    std::vector<float> multiplier;
    std::vector<float> tau_top;
    std::vector<float> tau_fluid;


    // are we hex_packed?
    bool hex_packed;
    bool three_series; // are we one clas of chips
    int initial_phase;
    //do we do this at all?
    bool do_xtalk_correction;
    bool simple_model;
    bool rescale_flag;

    // Stores json file fields, duplicates context_xxx fields
    // false if not a composite chip analysis
    bool if_block_analysis; 
    // xtalk parameters used for this region depend on absolute chip coordinates of the region
    int full_chip_x; 
    int full_chip_y; 
    // type of the chip being processed, and chip type in the loaded json (if any)
    std::string chipType; 
    std::string chipType_loaded; 


    void Allocate ( int size );
    void SetStandardGrid();
    void SetNewQuadGrid();
    void SetHexGrid();
    void SetAggressiveHexGrid();
    void SetNewHexGrid();
    void SetNewHexGridP0();

    // changed to json format 
    // void ReadCrossTalkFromFile ( const char * );

    // different chips unfortunately have different layouts
    void NeighborByGridPhase ( int &ncx, int &ncy, int cx, int cy, int cxd, int cyd, int phase );
    void NeighborByGridPhaseBB ( int &ncx, int &ncy, int cx, int cy, int cxd, int cyd, int phase );
    void NeighborByChipType ( int &ncx, int &ncy, int bead_rx, int bead_ry, int nei_idx, int region_x, int region_y );

    // this is effectively a constructor, loads paramters per region -- should we load xtalk.trace.json-s once and not for every region?
    void BootUpXtalkSpec( bool can_do_xtalk_correction, std::string &fname, std::string &_chipType, bool _if_block_analysis, int _full_chip_x, int _full_chip_y );
    TraceCrossTalkSpecification();

    void ReadCrossTalkFromFile( std::string &fname ); 
    void LoadJson(Json::Value &json, const std::string &fname);
    void UnpackTraceXtalkInfo(Json::Value &json);  
    void PackTraceXtalkInfo(Json::Value &json); 
    void SerializeJson(const Json::Value &json);
    // Current json loading process is irreversable if xtalk varies by region/by block
    // We cannot recunstruct full chip json structure for the output based on one loaded region
    // Instead, only parameters computed for this region are printed into a json format (compatible with full json format)
    // Alternatively, data stractures can be changed to accomodate the entire json (all regions), but it is probably not needed
    void TestWrite();

    // changed json format to support per-block parameters
    // redo the IO sensibly
    //void PackTraceXtalkInfo(Json::Value &json);
    //void SerializeJson(const Json::Value &json);
    //void LoadJson(Json::Value & json, const std::string& filename_json);
    //void WriteJson(const Json::Value & json, const std::string& filename_json);

  private:
    // Boost serialization support:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize ( Archive& ar, const unsigned int version )
    {
      fprintf ( stdout, "Serialize CrossTalkSpec ... " );
      ar
      & nei_affected
      & cx
      & cy
      & multiplier
      & tau_top
      & tau_fluid
      & hex_packed
      & three_series
          & initial_phase
      & do_xtalk_correction
      & simple_model
      & rescale_flag
      & if_block_analysis
      & full_chip_x
      & full_chip_y
      & chipType
      & chipType_loaded;
      fprintf ( stdout, "done\n" );
    }
};


#endif // CROSSTALKSPEC_H

