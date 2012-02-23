/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h> // for PATH_MAX
#include <sys/types.h>	// for S_ISREG()
#include <sys/stat.h>	// for stat()
#include <unistd.h> // for getcwd()
#include "TFs.h"
#include "Utils.h"

#include "dbgmem.h"

TFs::TFs(const char *_flowOrder)
{
	tfInfo = NULL;
	numTFs = 0;
	flowOrder = strdup(_flowOrder);
	numFlowsPerCycle = strlen(flowOrder);
	UpdateIonograms();
}

TFs::~TFs()
{
	if (tfInfo != NULL)
		free(tfInfo);
	if (flowOrder)
		free(flowOrder);
}

// Load - loads a TF config file
// expected format has fields comma-separated, and contains one entry per line as:
//    TF name, key, sequence
bool TFs::LoadConfig(char *file)
{
	char buf[1024];
	FILE *fp = fopen(file, "r");
	numTFs = 0;
	if (fp) {
		if (tfInfo != NULL)
			free(tfInfo);
		tfInfo = NULL;
		while (fgets(buf, sizeof(buf), fp) != NULL) {
			if ((buf[0] != '#') && (strlen(buf) > 10)) {
				// scan in the entry

				// read the name
				char name[64];
				char *comma = strchr(buf, ',');
				if (comma) {
					*comma = 0;
					comma++;
					strncpy(name, buf, sizeof(name));
					name[sizeof(name)-1] = 0; // keeps people from entering crazy-long names that could crash us

					// read the key
					char key[64];
					char *keyPtr = comma;
					// first skip to the 1st key char
					while (keyPtr != NULL && !isalpha(*keyPtr) && *keyPtr != 0) keyPtr++;
					comma = strchr(keyPtr, ','); // find the end
					*comma = 0; // NULL-out the end
					comma++;
					strncpy(key, keyPtr, sizeof(key));
					key[sizeof(key)-1] = 0; // keeps people from entering crazy-long keys that could crash us

					// the sequence 
					char seq[1024];
					sscanf(comma, "%s", seq);
					seq[sizeof(seq)-1] = 0;

					if (strlen(seq) > 0) {
						// allocate space for the new entry
						if (tfInfo == NULL)
							tfInfo = (TFInfo *)malloc(sizeof(TFInfo));
						else
							tfInfo = (TFInfo *)realloc(tfInfo, (numTFs+1)*sizeof(TFInfo));

						// save entry into our struct
						strcpy(tfInfo[numTFs].name, name);
						strcpy(tfInfo[numTFs].key, key);
						strcpy(tfInfo[numTFs].seq, seq);
						tfInfo[numTFs].len = strlen(seq);
						tfInfo[numTFs].count = 0;
						memset(tfInfo[numTFs].Ionogram, 0, sizeof(tfInfo[numTFs].Ionogram));
						tfInfo[numTFs].flows = 0;

						numTFs++;
					}
				}
			}
		}
		fclose(fp);
	}

	if (numTFs == 0) {
		//THIS IS THE NEW ERROR CONDITION TO HANDLE: NO TFs TO MATCH AGAINST
		//Should never get here.  Only if file exists but is empty.
		fprintf (stderr, "NO TFs TO USE!!!\n");
		return false;
	}

	UpdateIonograms();

	return true;
}

void TFs::UpdateIonograms()
{
	// convert each TF into array of flow-space hits
	char keyAndSeq[2048];
	int tf;
	for(tf=0;tf<numTFs;tf++) {
		snprintf(keyAndSeq, 512, "%s%s", tfInfo[tf].key, tfInfo[tf].seq);
		int keyAndSeqLen = strlen(keyAndSeq);
		tfInfo[tf].flows = GenerateIonogram(keyAndSeq, keyAndSeqLen, tfInfo[tf].Ionogram);
	}
}

int TFs::GenerateIonogram(const char *seq, int len, int *ionogram)
{
  return(seqToFlow(seq,len,ionogram,800,flowOrder,numFlowsPerCycle));
}

char *TFs::GetTFConfigFile ()
{
	return (GetIonConfigFile ("DefaultTFs.conf"));
}

/*******************************************************************************
**
**	TFTracker Class
**
*******************************************************************************/

TFTracker::TFTracker(char *experimentName)
{
	char file[] = {"TFTracking.txt"};
	fileName = (char *) malloc (strlen(experimentName) + strlen(file) + 2);
	sprintf (fileName, "%s/%s", experimentName, file);
	fd = NULL;
	fd = fopen_s(&fd, fileName, "wb");
	if (!fd)
		fprintf (stderr, "error creating TFTracking.txt (errno: #%d)\n", errno);

	fprintf (fd, "#Format: row,column,label\n");
}

TFTracker::~TFTracker()
{
    free(fileName);
}

bool TFTracker::Add(int row, int col, char *label)
{
	if (fd == NULL) {
		//	Cannot access file to write to
		return (true);
	}
	if (label)
		fprintf (fd, "%d,%d,%s\n",row,col,label);
	else
		fprintf (fd, "%d,%d,\n",row,col);
		   
	return (false);
}

bool TFTracker::Close()
{
	if (fd)
		fclose (fd);
		
	return (false);
}
