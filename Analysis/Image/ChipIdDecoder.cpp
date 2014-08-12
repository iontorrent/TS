/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ChipIdDecoder.h"
#include <stddef.h>
#include "Utils.h"

ChipIdEnum ChipIdDecoder::glob_chip_id = ChipIdUnknown;

ChipIdDecodeArrayType ChipIdDecoder::chip_id_str_lookup_array[] = 
{
  { "316", ChipId316    },
  { "316v2",  ChipId316v2    },
  { "318", ChipId318    },
  { "314", ChipId314    },
  { "p1.1.17", ChipId1_1_17    },
  { "p1.2.18", ChipId1_2_18    },
  { "p1.0.19", ChipId1_0_19    },
  { "p2.2.1", ChipId2_2_1    },
  { "900", ChipId_old_P1    },  //for backward compatibility  and basecalling of old old chips
  { "p1.0.20", ChipId1_0_20    },
  { NULL,   ChipIdUnknown },
};

char * ChipIdDecoder::chipType = NULL;

// sets a the static glob_chip_id.  The XTChannelCorrect method uses the chip id to determine which correction vector,
// if any, to apply to the data.
//@TODO: global variables are toxic please minimize their use
void ChipIdDecoder::SetGlobalChipId(const char *id_str_from_explog)
{
  // lookup the id string from the explog file and decode it into a simple enumeration
  if(id_str_from_explog != NULL)
  {
    if (chipType) { free(chipType); }
    chipType = strdup(id_str_from_explog); // generate a copy so we can use this everywhere
    ToLower(chipType);
    for(int nchip = 0;chip_id_str_lookup_array[nchip].id_str != NULL;nchip++)
    {
      if(strcmp(chip_id_str_lookup_array[nchip].id_str,chipType) == 0)
      {
        glob_chip_id = chip_id_str_lookup_array[nchip].id;
        printf("Found chip id %s\n",chipType);
        return;
      }
    }
  }

  printf("Unknown chip id str %s\n",id_str_from_explog);
}

// we want the text version sometimes as well
// as long as we are sinning with a global variable
char *ChipIdDecoder::GetChipType(void)
{
  return(chipType);
}

ChipIdDecoder::~ChipIdDecoder() {

  if (chipType != NULL){
    free (chipType);
    chipType = NULL;
  }
  
}

// Please do not explicitly use enumerated chip IDs outside of this structure
// that makes multiple places where decisions need to be visited based on chip type
// ideally, we want configuration files per chip to replace hardcoded decisions
bool ChipIdDecoder::IsProtonChip(){
  switch (glob_chip_id) {
    case ChipId1_1_17:
    case ChipId1_2_18:
    case ChipId2_2_1:
    case ChipId1_0_19:
    case ChipId1_0_20:
    case ChipId_old_P1:
      return true;
      break;
    default:
      return false;
  }
}

bool ChipIdDecoder::IsLargePGMChip(){
  switch (glob_chip_id) {
    //case ChipId314:
    case ChipId316:
    case ChipId316v2:
    case ChipId318:


      return true;
      break;
    default:
      return false;
  }
}


bool ChipIdDecoder::BigEnoughForGPU(){
  switch (glob_chip_id) {
    case ChipId318:
    case ChipId1_1_17:
    case ChipId1_0_19:
    case ChipId1_2_18:
    case ChipId1_0_20:
    case ChipId2_2_1:
    case ChipId_old_P1:
      return true;
    case ChipId314:
    case ChipId316:
    case ChipId316v2:
    default:
    {
      return false;
    }
  }
}

bool ChipIdDecoder::NeedsNighborPixelCorrection(){
    return (glob_chip_id==ChipId2_2_1);
}

std::string get_KnownAlternate_chiptype(std::string chiptype) {
    if (chiptype == "900")
        chiptype = "p1.1.17";
    return (chiptype);
}

std::string get_KnownAlternate_chiptype(const char *chiptype) {
    if (strcmp(chiptype,"900")==0)
        chiptype = "p1.1.17";
    return (chiptype);
}



