/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef HANDLEEXPLOG_H
#define HANDLEEXPLOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <fstream>
#include <errno.h>
#include <sys/types.h>
#include <math.h>
#include <unistd.h>
#include <vector>
#include <bitset>
#include <sstream>
#include <sys/time.h>

int   GetCycles (const char *filepath);
int   GetTotalFlows (const char *filepath);
char  *GetChipId (const char *filepath);

char    *GetExpLogParameter (const char *filename, const char *paramName);
void GetExpLogParameters (const char *filename, const char *paramName,
                          std::vector<std::string> &values);
char* MakeExpLogPathFromDatDir(const char *dat_source_directory);
char* MakeExpLogFinalPathFromDatDir(const char *dat_source_directory);

int     HasWashFlow (char *filepath);
char  *GetPGMFlowOrder (char *filepath);

bool ifDatacollectExcludeRegion(const char *filename, int beginX, int endX, int chipSizeX, int beginY, int endY, int chipSizeY);

#endif //HANDLEEXPLOG_H
