#ifndef __RSMAgent__
#define __RSMAgent__

#include "AeOSLocal.h"
#include "AeTypes.h"

extern void UpdateDataItem(AeDRMQueuePriority priority, AeDRMDataItem *dataItem);
void trimTrailingWhitespace(char *inputBuf);
void WriteAeAnalogDataItem(char const * const subcat, char const * const softwareComponent, 
	double value, AeDRMDataItem *item);
void WriteAeStringDataItem(char const * const subcat, char const * const softwareComponent, 
	char const * const version, AeDRMDataItem *item);

#endif // __RSMAgent__

