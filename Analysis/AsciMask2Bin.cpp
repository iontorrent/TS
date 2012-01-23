/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <string.h>
#include "Mask.h"
#include "Utils.h"

//#define MaskExclude (1<<10)

void Trim(char *buf)
{
	int len = strlen(buf);
	while (len > 0 && (buf[len-1] == '\r' || buf[len-1] == '\n'))
		len--;
	buf[len] = 0;
}

int main(int argc, char *argv[])
{
	if (argc != 1)
		return 1;
	
	int width = 0;
	int height = 0;
	char *rawName = argv[1];

	FILE *fp = fopen(rawName, "r");
	if (fp) {
		// determine the width & height from the file
		char buf[16384];
		while (fgets(buf, sizeof(buf), fp)) {
			Trim(buf); // remove trailing carriage returns, etc.
			if (buf[0] == '0' || buf[0] == '1') {
				if (width == 0)
					width = strlen(buf);
				height++;
			}
		}
		fseek(fp, 0, SEEK_SET);

		if (width > 0 && height > 0) {
			printf("Generating %dx%d excludeMask\n", width, height);
			Mask excludeMask(width, height);
			int x, y = 0;
			while (fgets(buf, sizeof(buf), fp)) {
				Trim(buf); // remove trailing carriage returns, etc.
				if (buf[0] == '0' || buf[0] == '1') {
					for(x=0;x<width;x++) {
						if (buf[x] == '1')
							excludeMask[x+y*width] = MaskExclude;
					}
					y++;
				}
			}
			char maskName[MAX_PATH_LENGTH];
			strcpy(maskName, rawName);
			char *ptr = strrchr(maskName, '.');
			if (ptr) {
				strcpy(ptr+1, "bin");
			} else {
				strcat(maskName, ".bin");
			}

			printf("Writing out mask: %s\n", maskName);
			excludeMask.WriteRaw(maskName);
		}
		fclose(fp);
	}
}

