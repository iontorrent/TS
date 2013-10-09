/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ChipIdDecoder.h"

ChipIdEnum ChipIdDecoder::glob_chip_id = ChipIdUnknown;

ChipIdDecodeArrayType ChipIdDecoder::chip_id_str_lookup_array[] = 
{
  { "316", ChipId316    },
  { "316v2",  ChipId316v2    },
  { "318", ChipId318    },
  { "314", ChipId314    },
  { "900", ChipId900    },
  { "910", ChipId910    },
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
    for(int nchip = 0;chip_id_str_lookup_array[nchip].id_str != NULL;nchip++)
    {
      if(strcmp(chip_id_str_lookup_array[nchip].id_str,id_str_from_explog) == 0)
      {
          glob_chip_id = chip_id_str_lookup_array[nchip].id;
          printf("Found chip id %s\n",id_str_from_explog);
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
