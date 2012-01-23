/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

const char *validate_fmt(const char* fmt, const char* file, int32_t line)
{
	//Unused parameter generates compiler warning, so...
	if (file) {};
	if (line) {};
	
	return fmt;
/*
	const char* p = fmt;
	while (1) {
		p = strstr( p, "%s" );
		if (p == NULL) break;
		if ((p == fmt) || (*(p-1) != '%')) {
			fprintf(stderr, "Hey, you used \"%%s\" in %s: line %d!\n", file, line);
			abort();
		}
	}
	return fmt;
*/
}

const char *validate_str(const char *src, int destsize, const char* file, int32_t line)
{
	int len = strlen(src);
	if (destsize <= len) {
		fprintf(stderr, "ERROR!  Destination string not large enough for operation in %s: line %d!\n", file, line);
		abort();
	}
	return src;
}

