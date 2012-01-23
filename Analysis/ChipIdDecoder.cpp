/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ChipIdDecoder.h"

ChipIdEnum ChipIdDecoder::glob_chip_id = ChipIdUnknown;

ChipIdDecodeArrayType ChipIdDecoder::chip_id_str_lookup_array[] = 
{
  { "316", ChipId316    },
  { "318", ChipId318    },
  { "314", ChipId314    },
  { "900", ChipId900    },
  { NULL,   ChipIdUnknown },
};

// sets a the static glob_chip_id.  The XTChannelCorrect method uses the chip id to determine which correction vector,
// if any, to apply to the data.
void ChipIdDecoder::SetGlobalChipId(const char *id_str_from_explog)
{
  // lookup the id string from the explog file and decode it into a simple enumeration
  if(id_str_from_explog != NULL)
  {
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

