// for strdup
#define _BSD_SOURCE

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "list.h"

typedef struct host_s
{
	char *name;
	int port;
} host_t;

static host_t *allocateHost(char *name, int port)
{
	host_t *host = malloc(sizeof(host_t));
	if (!host) {
		fprintf(stderr, "%s: out of memory.\n", __FUNCTION__);
		return NULL;
	}
	host->name = NULL;
	if (name)
		host->name = strdup(name);
	host->port = port;

	return host;
}

void freeHost(host_t *host)
{
	if (!host)
		return;
	if (host->name) {
		free(host->name);
		host->name = NULL;
	}
	host->port = 0;
	free(host);
}

void readHostList(char const * const listFile, list_t *hosts)
{
	if (!listFile) {
		printf("%s: NULL listFile arg\n", __FUNCTION__);
		return;
	}

	FILE *fp;
	fp = fopen(listFile, "r");
	if (fp) {
#define LINESIZE 256
		char line[LINESIZE] = {0};
		while (fgets(line, LINESIZE, fp)) {

			// Ignore comments
			char *pos = strchr(line, '#');
			if (pos)
				*pos = '\0';
			// Skip empty lines and comment-only lines.
			if (strlen(line) == 0)
				continue;


			// Trim input lines.
			pos = &line[strlen(line) - 1];
			while (pos >= line && isspace(*pos))
				*pos-- = '\0';
			// Skip all-blank lines.
			if (strlen(line) == 0)
				continue;

			// parse hostname and port number from line.
			char *name = line;
			int port = 22;
			pos = strchr(line, ' ');
			if (pos) {
				port = atoi(pos);
				*pos = '\0';
			}

			// Store input lines
			list_append(hosts, allocateHost(name, port));
		}
		if (fclose(fp))
			printf("%s: fclose: %s\n", __FUNCTION__, strerror(errno));
	}
	else {
		hosts->item = allocateHost("rssh.iontorrent.net", 22);
	}
}

#ifdef READHOSTLIST_UNITTEST
int main(void)
{
	char const * const listFile = "rsshHosts.conf";
	list_t *hosts = list_create();
	readHostList(listFile, hosts);

	list_t *hptr = hosts;
	while (hptr) {
		host_t *h = (host_t *)(hptr->item);
		if (h)
			printf("%s port %d\n", h->name, h->port);
		hptr = list_nextItem(hptr);
	}

	hptr = hosts;
	while (hptr) {
		freeHost(hptr->item);
		hptr = hptr->next;
	}
	list_free(hosts);

	return 0;
}
#endif
