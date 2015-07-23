#ifndef READ_HOST_LIST_H
#define READ_HOST_LIST_H

#include "list.h"

typedef struct host_s
{
	char *name;
	int port;
} host_t;

void readHostList(char const * const listFile, list_t *hosts);
void freeHost(host_t *host);

#endif	// READ_HOST_LIST_H
