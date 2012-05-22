/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "MaskFunctions.h"




void ExportSubRegionSpecsToMask (SpatialContext &loc_context)
{
  // Default analysis mode sets values to 0 and whole-chip processing proceeds.
  // otherwise, command line override (--analysis-region) can define a subchip region.
  Mask::chipSubRegion.row = (loc_context.GetChipRegion()).row;
  Mask::chipSubRegion.col = (loc_context.GetChipRegion()).col;
  Mask::chipSubRegion.h = (loc_context.GetChipRegion()).h;
  Mask::chipSubRegion.w = (loc_context.GetChipRegion()).w;
}


void UpdateBeadFindOutcomes (Mask *maskPtr, Region &wholeChip, char *experimentName, bool not_single_beadfind, int update_stats)
{
  char maskFileName[2048];
  if (!update_stats)
  {
    sprintf (maskFileName, "%s/bfmask.stats", experimentName);
    //maskPtr->DumpStats (wholeChip, maskFileName, !clo.bfd_control.SINGLEBF);
    maskPtr->DumpStats (wholeChip, maskFileName, not_single_beadfind);
  }
  // analysis.bfmask.bin is what BaseCaller expects
  sprintf (maskFileName, "%s/analysis.bfmask.bin", experimentName);
  maskPtr->WriteRaw (maskFileName);
  maskPtr->validateMask();
}


void LoadBeadMaskFromFile (SystemContext &sys_context,  Mask *maskPtr)
{
  char maskFileName[2048];

  // Load beadmask from file
  sprintf (maskFileName, "%s/%s",sys_context.wellsFilePath, "./bfmask.bin");

  maskPtr->SetMask (maskFileName);
  if (maskPtr->SetMask (maskFileName))
  {
    exit (EXIT_FAILURE);
  }

}


//@TODO: Side effects!!!!  Munges command line opts randomly

void SetSpatialContextAndMask(SpatialContext &loc_context, Mask *maskPtr, int &rows, int &cols)
{
  rows = maskPtr->H();
  cols = maskPtr->W();
  
  loc_context.rows = rows;
  loc_context.cols = cols;

  //--- Note that if we specify cropped regions on the command line, we are supposing that the original
  //    analysis was a whole chip analysis.  This is a safe assumption for the most part.
  // TODO: Need to rationalize the cropped region handling.

  loc_context.regionsX = 1;
  loc_context.regionsY = 1;

  if (loc_context.numRegions == 1)
  {
    int rx, ry;
    for (ry = 0; ry < rows; ry++)
    {
      for (rx = 0; rx < cols; rx++)
      {
        if (rx >= loc_context.regionXOrigin && rx < (loc_context.regionXOrigin
            + loc_context.regionXSize) && ry >= loc_context.regionYOrigin && ry
            < (loc_context.regionYOrigin +loc_context.regionYSize))
          ;
        else
          (*maskPtr) [rx + ry * cols] = MaskExclude;
      }
    }
  }


  //--- Handle cropped regions defined from command line
  if (loc_context.numCropRegions > 0)
  {
    maskPtr->CropRegions (loc_context.cropRegions,loc_context.numCropRegions, MaskExclude);
  }
}

  //Uncomment next line to revert to old exclusion mask file usage.  Remove once it is passed.
#define OLDWAY

//@TODO:  Please do not use inline #ifdefs to rewrite code
// this should only be done >at the function level< to maintain readability.
// if you believe the function is too long to do this, >then rewrite the function to be shorter or more than one piece<.


#ifdef OLDWAY

void SetExcludeMask (CommandLineOpts &clo, Mask *maskPtr, char *chipType, int rows, int cols)
{

  bool applyExclusionMask = true;

  /*
   *  Determine if this is a cropped dataset
   *  3 types:
   *    wholechip image dataset - rows,cols should be == to chip_len_x,chip_len_y
   *    cropped image dataset - above test is false AND chip_offset_x == -1
   *    blocked image dataset - above test is false AND chip_offset_x != -1
   */
  if ( (rows == clo.loc_context.chip_len_y) && (cols == clo.loc_context.chip_len_x))
  {
    //This is a wholechip dataset
    applyExclusionMask = true;
    fprintf (stderr, "This is a wholechip dataset so the exclusion mask will be applied\n");
  }
  else
  {
    if (clo.loc_context.chip_offset_x == -1)
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
  if (clo.loc_context.numCropRegions == 0)
  {
    maskPtr->Init (cols, rows, MaskEmpty);
  }
  else
  {
    maskPtr->Init (cols, rows, MaskExclude);

    // apply one or more cropRegions, mark them MaskEmpty
    for (int q = 0; q < clo.loc_context.numCropRegions; q++)
    {
      maskPtr->MarkRegion (clo.loc_context.cropRegions[q], MaskEmpty);
    }
  }

  /*
   * Apply exclude mask from file
   */
  clo.loc_context.exclusionMaskSet = false;
  if (chipType && applyExclusionMask)
  {
    char *exclusionMaskFileName = NULL;
    char filename[64] = { 0 };

    sprintf (filename, "exclusionMask_%s.bin", chipType);

    exclusionMaskFileName = GetIonConfigFile (filename);
    fprintf (stderr, "Exclusion Mask File = '%s'\n", exclusionMaskFileName);
    if (exclusionMaskFileName)
    {
      clo.loc_context.exclusionMaskSet = true;

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

void SetExcludeMask (CommandLineOpts &clo, Mask *maskPtr, char *chipType, int rows, int cols)
{

  bool applyExclusionMask = true;

  /*
   *  Determine if this is a cropped dataset
   *  3 types:
   *    wholechip image dataset - rows,cols should be == to chip_len_x,chip_len_y
   *    cropped image dataset - above test is false AND chip_offset_x == -1
   *    blocked image dataset - above test is false AND chip_offset_x != -1
   */
  if ( (rows == clo.loc_context.chip_len_y) && (cols == clo.loc_context.chip_len_x))
  {
    //This is a wholechip dataset
    applyExclusionMask = true;
    fprintf (stderr, "This is a wholechip dataset so the exclusion mask will be applied\n");
  }
  else
  {
    if (clo.loc_context.chip_offset_x == -1)
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
  if (clo.loc_context.numCropRegions == 0)
  {
    maskPtr->Init (cols, rows, MaskEmpty);
  }
  else
  {
    maskPtr->Init (cols, rows, MaskExclude);

    // apply one or more cropRegions, mark them MaskEmpty
    for (int q = 0; q < clo.loc_context.numCropRegions; q++)
    {
      maskPtr->MarkRegion (clo.loc_context.cropRegions[q], MaskEmpty);
    }
  }

  /*
   * Apply exclude mask from file
   */
  clo.loc_context.exclusionMaskSet = false;
  if (chipType && applyExclusionMask)
  {
    char *exclusionMaskFileName = NULL;
    char filename[64] = { 0 };

    sprintf (filename, "excludeMask_%s", chipType);

    exclusionMaskFileName = GetIonConfigFile (filename);
    fprintf (stderr, "Exclusion Mask File = '%s'\n", exclusionMaskFileName);
    if (exclusionMaskFileName)
    {
      clo.loc_context.exclusionMaskSet = true;

      FILE *excludeFile = NULL;
      excludeFile = fopen (exclusionMaskFileName,"rb");
      assert (excludeFile != NULL);
      uint16_t x = 0;
      uint16_t y = 0;
      while (1)
      {
        if (fread (&x, sizeof (x), 1, excludeFile) != 1) break;
        if (fread (&y, sizeof (y), 1, excludeFile) != 1) break;
        //fprintf (stderr, "Excluding %d %d (%d %d)\n",x,y,(int) x - clo.chip_offset_x,(int) y - clo.chip_offset_y);
        maskPtr->Set ( (int) x - clo.loc_context.chip_offset_x, (int) y - clo.loc_context.chip_offset_y,MaskExclude);
      }

    }
    else
    {
      fprintf (stderr, "WARNING: Exclusion Mask %s not applied\n", filename);
    }
  }
}

#endif 
