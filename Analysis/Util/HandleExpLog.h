/* Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef HANDLEEXPLOG_H
#define HANDLEEXPLOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <math.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <sys/time.h>

int   GetCycles (const char *filepath);
int   GetTotalFlows (const char *filepath);
char  *GetChipId (const char *filepath);

char    *GetExpLogParameter (const char *filename, const char *paramName);
void GetExpLogParameters (const char *filename, const char *paramName,
                          std::vector<std::string> &values);
char* MakeExpLogPathFromDatDir(const char *dat_source_directory);
int     HasWashFlow (char *filepath);
char  *GetPGMFlowOrder (char *filepath);


#endif //HANDLEEXPLOG_H
