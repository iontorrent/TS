/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "MaskFunctions.h"



//@TODO:  Please do not use inline #ifdefs to rewrite code
// this should only be done >at the function level< to maintain readability.
// if you believe the function is too long to do this, >then rewrite the function to be shorter or more than one piece<.

 //Uncomment next line to revert to old exclusion mask file usage.  Remove once it is passed.
#define OLDWAY

#ifdef OLDWAY

void SetExcludeMask (SpatialContext &loc_context, Mask *maskPtr, char *chipType, int rows, int cols)
{

  bool applyExclusionMask = true;

  /*
   *  Determine if this is a cropped dataset
   *  3 types:
   *    wholechip image dataset - rows,cols should be == to chip_len_x,chip_len_y
   *    cropped image dataset - above test is false AND chip_offset_x == -1
   *    blocked image dataset - above test is false AND chip_offset_x != -1
   */
  if ( (rows == loc_context.chip_len_y) && (cols == loc_context.chip_len_x))
  {
    //This is a wholechip dataset
    applyExclusionMask = true;
    fprintf (stderr, "This is a wholechip dataset so the exclusion mask will be applied\n");
  }
  else
  {
    if (loc_context.chip_offset_x == -1)
    {
      applyExclusionMask = false;
      fprintf (stderr, "This is a cropped dataset so the exclusion mask will not be applied\n");
    }
    else
    {
      applyExclusionMask = false;
      fprintf (stderr, "This is a block dataset so the exclusion mask will not be applied\n");

    }
  }
  /*
   *  If we get a cropped region definition from the command line, we want the whole chip to be MaskExclude
   *  except for the defined crop region(s) which are marked MaskEmpty.  If no cropRegion defined on command line,
   *  then we proceed with marking the entire chip MaskEmpty
   */
  if (loc_context.numCropRegions == 0)
  {
    maskPtr->Init (cols, rows, MaskEmpty);
  }
  else
  {
    maskPtr->Init (cols, rows, MaskExclude);

    // apply one or more cropRegions, mark them MaskEmpty
    for (int q = 0; q < loc_context.numCropRegions; q++)
    {
      maskPtr->MarkRegion (loc_context.cropRegions[q], MaskEmpty);
    }
  }

  /*
   * Apply exclude mask from file
   */
  loc_context.exclusionMaskSet = false;
  if (chipType && applyExclusionMask)
  {
    char *exclusionMaskFileName = NULL;
    char filename[64] = { 0 };

    sprintf (filename, "exclusionMask_%s.bin", chipType);

    exclusionMaskFileName = GetIonConfigFile (filename);
    fprintf (stderr, "Exclusion Mask File = '%s'\n", exclusionMaskFileName);
    if (exclusionMaskFileName)
    {
      loc_context.exclusionMaskSet = true;

      Mask excludeMask (1, 1);
      excludeMask.SetMask (exclusionMaskFileName);
      free (exclusionMaskFileName);
      //--- Mark beadfind masks with MaskExclude bits from exclusionMaskFile
      maskPtr->SetThese (&excludeMask, MaskExclude);

    }
    else
    {
      fprintf (stderr, "WARNING: Exclusion Mask %s not applied\n", filename);
    }
  }
}

#else

void SetExcludeMask (SpatialContext &loc_context, Mask *maskPtr, char *chipType, int rows, int cols)
{

  bool applyExclusionMask = true;

  /*
   *  Determine if this is a cropped dataset
   *  3 types:
   *    wholechip image dataset - rows,cols should be == to chip_len_x,chip_len_y
   *    cropped image dataset - above test is false AND chip_offset_x == -1
   *    blocked image dataset - above test is false AND chip_offset_x != -1
   */
  if ( (rows == loc_context.chip_len_y) && (cols == loc_context.chip_len_x))
  {
    //This is a wholechip dataset
    applyExclusionMask = true;
    fprintf (stderr, "This is a wholechip dataset so the exclusion mask will be applied\n");
  }
  else
  {
    if (loc_context.chip_offset_x == -1)
    {
      applyExclusionMask = false;
      fprintf (stderr, "This is a cropped dataset so the exclusion mask will not be applied\n");
    }
    else
    {
      applyExclusionMask = true;
      fprintf (stderr, "This is a block dataset so the exclusion mask will be applied\n");
    }
  }
  /*
   *  If we get a cropped region definition from the command line, we want the whole chip to be MaskExclude
   *  except for the defined crop region(s) which are marked MaskEmpty.  If no cropRegion defined on command line,
   *  then we proceed with marking the entire chip MaskEmpty
   */
  if (loc_context.numCropRegions == 0)
  {
    maskPtr->Init (cols, rows, MaskEmpty);
  }
  else
  {
    maskPtr->Init (cols, rows, MaskExclude);

    // apply one or more cropRegions, mark them MaskEmpty
    for (int q = 0; q < loc_context.numCropRegions; q++)
    {
      maskPtr->MarkRegion (loc_context.cropRegions[q], MaskEmpty);
    }
  }

  /*
   * Apply exclude mask from file
   */
  loc_context.exclusionMaskSet = false;
  if (chipType && applyExclusionMask)
  {
    char *exclusionMaskFileName = NULL;
    char filename[64] = { 0 };

    sprintf (filename, "excludeMask_%s", chipType);

    exclusionMaskFileName = GetIonConfigFile (filename);
    fprintf (stderr, "Exclusion Mask File = '%s'\n", exclusionMaskFileName);
    if (exclusionMaskFileName)
    {
      loc_context.exclusionMaskSet = true;

      FILE *excludeFile = NULL;
      excludeFile = fopen (exclusionMaskFileName,"rb");
      assert (excludeFile != NULL);
      uint16_t x = 0;
      uint16_t y = 0;
      while (1)
      {
        if (fread (&x, sizeof (x), 1, excludeFile) != 1) break;
        if (fread (&y, sizeof (y), 1, excludeFile) != 1) break;
        //fprintf (stderr, "Excluding %d %d (%d %d)\n",x,y,(int) x - chip_offset_x,(int) y - chip_offset_y);
        maskPtr->Set ( (int) x - loc_context.chip_offset_x, (int) y - loc_context.chip_offset_y,MaskExclude);
      }

    }
    else
    {
      fprintf (stderr, "WARNING: Exclusion Mask %s not applied\n", filename);
    }
  }
}

#endif 
