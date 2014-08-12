/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RSMAGENT_H
#define RSMAGENT_H

#include "AeOSLocal.h"
#include "AeTypes.h"

extern void UpdateDataItem(AeDRMQueuePriority priority, AeDRMDataItem *dataItem);
void trimTrailingWhitespace(char *inputBuf);
void WriteAeAnalogDataItem(char const * const subcat, char const * const softwareComponent, 
	double value, AeDRMDataItem *item);
void WriteAeStringDataItem(char const * const subcat, char const * const softwareComponent, 
	char const * const version, AeDRMDataItem *item);

#endif // RSMAGENT_H

