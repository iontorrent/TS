/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "MaskFunctions.h"

void SetExcludeMask(SpatialContext& loc_context, Mask* maskPtr, char* chipType,
    int rows, int cols, std::string exclusionMaskFile,
    bool isThumbnail)
{
    bool applyExclusionMask = (isThumbnail) ? false : true;

    /*
     *  If we get a cropped region definition from the command line, we want the
     * whole chip to be MaskExclude
     *  except for the defined crop region(s) which are marked MaskEmpty.  If no
     * cropRegion defined on command line,
     *  then we proceed with marking the entire chip MaskEmpty
     *
     */

    if ( loc_context.isCropped ) {
        maskPtr->Init(cols, rows, MaskExclude);
    } else {
        maskPtr->Init(cols, rows, MaskEmpty);
    }

    if (applyExclusionMask && !exclusionMaskFile.empty()) {

            // apply one or more cropRegions, mark them MaskEmpty
            for (int q = 0; q < loc_context.numCropRegions; q++) {
                maskPtr->MarkRegion(loc_context.cropRegions[q], MaskEmpty);
            }

        /*
         * Apply exclude mask from file
         */
        loc_context.exclusionMaskSet = false;

        char* exclusionMaskFileName = GetIonConfigFile(exclusionMaskFile.c_str());

        fprintf(stderr, "Exclusion Mask File = '%s'\n", exclusionMaskFileName);

        if (exclusionMaskFileName) {
            loc_context.exclusionMaskSet = true;

            Mask excludeMask(loc_context.cols, loc_context.rows);

            if (exclusionMaskFile.compare(exclusionMaskFile.size() - 3, 3, "txt") == 0) { // this is text mask (e.g. for proton/S5)
                std::cerr << "Using text exclusion mask\n";

                // SpatialContext sets offsets to -1 if it can't parse them from the working directory
                // We, in principle, want to enable handling of a PGM .txt mask
                excludeMask.SetMaskFullChipText( exclusionMaskFileName,
                                                 max(0,loc_context.chip_offset_x), max(0,loc_context.chip_offset_y),
                                                 loc_context.cols, loc_context.rows );

                //--- Mark beadfind masks with MaskExclude bits from exclusionMaskFile
                maskPtr->SetThese(&excludeMask, MaskExclude);
            } else { // this is binary mask
                std::cerr << "Using binary exclusion mask\n";
                excludeMask.SetMask(exclusionMaskFileName);

                //--- Mark beadfind masks with MaskExclude bits from exclusionMaskFile
                maskPtr->SetThese(&excludeMask, MaskExclude);
            }
        } else if (chipType) {
            std::cerr << "Using binary exclusion mask based on chip type: "
                      << chipType << "\n";

            char filename[64] = { 0 };
            sprintf(filename, "exclusionMask_%s.bin", chipType);

            exclusionMaskFileName = GetIonConfigFile(filename);
            fprintf(stderr, "Exclusion Mask File = '%s'\n", exclusionMaskFileName);
            if (exclusionMaskFileName) {
                loc_context.exclusionMaskSet = true;

                Mask excludeMask(1, 1);
                excludeMask.SetMask(exclusionMaskFileName);

                //--- Mark beadfind masks with MaskExclude bits from exclusionMaskFile
                maskPtr->SetThese(&excludeMask, MaskExclude);

            } else {
                fprintf(stderr, "WARNING: Exclusion Mask %s not applied\n", filename);
            }
        }
        free(exclusionMaskFileName);
    } else {
        std::cerr << "Exclusion mask is not applied";
    }
}
