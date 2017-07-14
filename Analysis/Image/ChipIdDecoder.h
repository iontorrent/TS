/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef CHIPIDDECODER_H
#define CHIPIDDECODER_H

#include <string>

typedef enum
{
  ChipIdUnknown = 0,
  ChipId314 = 1,
  ChipId316 = 2,
  ChipId316v2 = 3,        
  ChipId318 = 4,
  ChipId1_1_17  = 5,
  ChipId1_0_19  = 6,
  ChipId2_2_1  = 7,
  ChipId2_2_2  = 8,
  ChipId1_2_18  = 9,
  ChipId_old_P1 = 10,
  ChipId1_0_20  = 11,
  ChipId550 = 12,
  ChipId540 = 13,
  ChipId530 = 14,
  ChipId520 = 15,
  ChipId551 = 16,
  ChipId541 = 17,
  ChipId531 = 18,
  ChipId521 = 19,
  ChipId2_0_1 = 20,
  ChipId2_1_1 = 21,
  ChipId2_3_1 = 22,
  ChipId1_1_541 = 23,
  ChipId522 = 24, 
  ChipId560 = 25
} ChipIdEnum;

typedef struct
{
  const char *id_str;
  ChipIdEnum id;
} ChipIdDecodeArrayType;

class ChipIdDecoder
{

  public:
    static ChipIdEnum GetGlobalChipId (void) { return glob_chip_id; }
    static void SetGlobalChipId (const char* id_str_from_explog);
    static char *GetChipType(void);
    static bool IsProtonChip();
    static bool IsPzero();
    static bool IsPone();
    static bool IsPtwo();
    static bool IsChip314();
    static bool IsChip316();
    static bool IsChip316v2();
    static bool IsChip318();
    static bool IsLargePGMChip();
    static bool BigEnoughForGPU();
    static bool NeedsNighborPixelCorrection();

    ~ChipIdDecoder();
    ChipIdDecoder() { chipType = NULL; }
    
  private:
    static ChipIdDecodeArrayType chip_id_str_lookup_array[];
    static ChipIdEnum glob_chip_id;
    static char *chipType;
};

std::string get_KnownAlternate_chiptype(std::string chiptype);
std::string get_KnownAlternate_chiptype(const char *chiptype);

#endif // CHIPIDDECODER_H
