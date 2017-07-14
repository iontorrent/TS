/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "SpatialContext.h"
#include "HandleExpLog.h"
#include "ImageTransformer.h"
#include "ChipIdDecoder.h"

void SpatialContext::DefaultSpatialContext()
{
  numRegions = 0;
  /* enables sub chip analysis */
  chipRegion.row=0;
  chipRegion.col=0;
  chipRegion.w=0;
  chipRegion.h=0;
  cols = 0;
  rows = 0;
  // regionXSize = 50;
  // regionYSize = 50;
  regionXSize = 0;
  regionYSize = 0;
  cropRegions = NULL;
  numCropRegions = 0;
  isCropped = false;
  // some raw image processing (like cross talk correction in the Image class) needs the absolute coordinates of the
  // pixels in the image.  This is easy for a standard data set, but for a cropped data set the origin of the data is
  // unknown.  These allow the user to specify the location of the cropped region so that these parts of analysis
  // will work as designed.
  cropped_region_x_offset = 0;
  cropped_region_y_offset = 0;

  // datasets that are divided into blocks; each block has an offset from the chip's origin:
  chip_offset_x = -1;
  chip_offset_y = -1;
  chip_len_x = 0;
  chip_len_y = 0;
  percentEmptiesToKeep = 100;
  exclusionMaskSet = false;
  regionsX = 0;
  regionsY = 0;
}

SpatialContext::~SpatialContext()
{
  if (cropRegions)
    delete (cropRegions);
}

void SpatialContext::FindDimensionsByType (char *explog_path)
{
  int dims[2];
  GetChipDim (chipType.c_str(), dims, explog_path); // @what if we're doing from wells and there are no dats?
  chip_len_x = dims[0];
  chip_len_y = dims[1];
}

//@TODO: these are really methods for location context, but...
//@TODO there are some weird effects in them that I don't like


//@TODO: for example, doesn't loc_context contain rows & cols?
void FixCroppedRegions (SpatialContext &loc_context, int _rows, int _cols)
{
  //If no cropped regions defined on command line, set cropRegions to whole chip
  if (!loc_context.cropRegions)
  {
    loc_context.numCropRegions = 1;
    loc_context.cropRegions = new Region(0, 0, _cols, _rows);
  }
}

//@TODO: doesn't loc_context >contain< rows & cols?
void SetUpRegionDivisions (SpatialContext &loc_context, int _rows, int _cols)
{
  int xinc, yinc;

  loc_context.regionsX = 1;
  loc_context.regionsY = 1;

  // fixed region size
  xinc = loc_context.regionXSize;
  yinc = loc_context.regionYSize;
  loc_context.regionsX = _cols / xinc;
  loc_context.regionsY = _rows / yinc;
  // make sure we cover the edges in case rows/yinc or cols/xinc not exactly divisible
  if ( ( (double) _cols / (double) xinc) != loc_context.regionsX)
    loc_context.regionsX++;
  if ( ( (double) _rows / (double) yinc) != loc_context.regionsY)
    loc_context.regionsY++;
  loc_context.numRegions = loc_context.regionsX * loc_context.regionsY;
}


void SetUpRegionsForAnalysis (int _rows, int _cols, SpatialContext &loc_context)
{
  FixCroppedRegions (loc_context, _rows, _cols);
  SetUpRegionDivisions (loc_context,_rows,_cols);
}

void SpatialContext::PrintHelp()
{
	printf ("     SpatialContext\n");
    printf ("     --region-size           INT VECTOR OF 2   setup region size in x and y []\n");
    printf ("     --cropped               INT VECTOR OF 4   setup cropped region in col, row, width and height []\n");
    printf ("     --analysis-region       INT VECTOR OF 4   setup chip region in col, row, width and height []\n");
    printf ("     --cropped-region-origin INT VECTOR OF 2   setup cropped region offset in x and y []\n");
    printf ("\n");
}

void SpatialContext::SetOpts(OptArgs &opts, Json::Value& json_params)
{
    chipType = GetParamsString(json_params, "chipType", "");
	vector<int> vec1;
	RetrieveParameterVectorInt(opts, json_params, '-', "region-size", "", vec1);
	if(vec1.size() > 0)
	{
		if(vec1.size() == 2)
		{
			regionXSize = vec1[0];
			regionYSize = vec1[1];
		}
		else
		{
			fprintf ( stderr, "Option Error: region-size format wrong, not size = 2\n" );
			exit ( EXIT_FAILURE );
		}
	}

	vector<int> vec2;
	RetrieveParameterVectorInt(opts, json_params, '-', "cropped", "", vec2);
	if(vec2.size() > 0)
	{
		if(vec2.size() == 4)
		{
			numCropRegions++;
			cropRegions = ( Region * ) realloc ( cropRegions, sizeof ( Region ) * numCropRegions );

			cropRegions[numCropRegions-1].col = vec2[0];
			cropRegions[numCropRegions-1].row = vec2[1];
			cropRegions[numCropRegions-1].w = vec2[2];
			cropRegions[numCropRegions-1].h = vec2[3];
            isCropped = true;
		}
		else
		{
			fprintf ( stderr, "Option Error: cropped format wrong, not size = 4\n" );
			exit ( EXIT_FAILURE );
		}
	}
		
	vector<int> vec3;
	RetrieveParameterVectorInt(opts, json_params, '-', "analysis-region", "", vec3);
	if(vec3.size() > 0)
	{
		if(vec3.size() == 4)
		{
			chipRegion.col = vec3[0];
			chipRegion.row = vec3[1];
			chipRegion.w = vec3[2];
			chipRegion.h = vec3[3];
		}
		else
		{
			fprintf ( stderr, "Option Error: analysis-region format wrong, not size = 4\n" );
			exit ( EXIT_FAILURE );
		}
	}

	vector<int> vec4;
	RetrieveParameterVectorInt(opts, json_params, '-', "cropped-region-origin", "", vec4);
	if(vec4.size() > 0)
	{
		if(vec4.size() == 2)
		{
			cropped_region_x_offset = vec4[0];
			cropped_region_y_offset = vec4[1];
		}
		else
		{
			fprintf ( stderr, "Option Error: cropped-region-origin format wrong, not size = 2\n" );
			exit ( EXIT_FAILURE );
		}
	}
}
