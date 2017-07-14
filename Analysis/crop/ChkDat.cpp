/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 * ChkDat.cpp
 *
 *  Created on: Jan 31, 2012
 *      Author: mbeauchemin
 */

/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/vfs.h>
#endif
#ifdef __APPLE__
#include <sys/uio.h>
#include <sys/mount.h>
#endif
#include <errno.h>
#include <assert.h>
#include "ByteSwapUtils.h"
#include "datahdr.h"
#include "LinuxCompat.h"
// #include "Raw2Wells.h"
#include "Image.h"
#include "crop/Acq.h"
#include "IonVersion.h"

#include <iostream>
using namespace std;

uint32_t numThreads=12;
uint32_t numDirs = 0;
uint32_t numAcq = 0;
int dont_wait = 0;
char *DstDir  = (char *)".";
char *oneFile = NULL;
uint32_t verbose=0;
uint32_t Error = 0;
char  DirList[200][512];


void GetFileName(uint32_t idx, uint32_t num, char *fname, int len)
{
	if(num == 0)
	{
		sprintf(fname,"%s/%s/beadfind_pre_0001.dat",DstDir,DirList[idx]);
	}
	else if(num == 1)
	{
		sprintf(fname,"%s/%s/beadfind_pre_0003.dat",DstDir,DirList[idx]);
	}
	else if(num < numAcq)
	{
		// acq
		sprintf(fname,"%s/%s/acq_%04d.dat",DstDir,DirList[idx],num-2);
	}
	else
	{
		sprintf(fname,"%s/%s/explog_final.txt",DstDir,DirList[idx]);
	}

}


void ChkFile(char *fname, size_t maxX=0, size_t maxY=0, size_t maxF=0, bool verbose=false)
{
	Image loader;

	if (loader.LoadRaw(fname))
	{
        // this one is ok
        //printf("File %s OK\n",fname);
        cout << fname << " rows=" << loader.GetRows() << " cols=" << loader.GetCols() << " fames=" << loader.GetFrames() << endl << flush;
        const RawImage *raw = loader.GetImage();

        if (verbose)
        {
            size_t frames = loader.GetFrames();
            size_t rows = loader.GetRows();
            size_t cols = loader.GetCols();
            size_t stride = rows*cols;
            cout << "flow\tx(col)\ty(row)\tvalue" << endl << flush;
            for (size_t f = 0; f < frames; f++) {
                if (maxF>0 && f>=maxF)
                    break;
              // First image copied in for this frame
              for (size_t r = 0; r < rows; r++) {
                  if (maxY>0 && r>=maxY)
                      break;
                for (size_t c = 0; c < cols; c++) {
                    if (maxX>0 && c>=maxX)
                        break;
                  cout << f << "\t" << c << "\t" << r << "\t" << raw->image[r * cols + c + stride * f] << endl << flush;
                }
              }
            }
        }
        loader.Close();
    }
	else
	{
		printf("File %s FAILED\n",fname);
		Error=1;
		exit(-1);
	}

}

void *worker(void *arg)
{
	uint64_t threadNum = (uint64_t)arg;
	uint32_t startIdx  = (numDirs*threadNum)/numThreads;
	uint32_t endIdx;
//	uint32_t beadfind3=0;
//	uint32_t beadfind1=0;
//	uint32_t acqNum=0;
	struct stat statBuf;
	uint32_t numFiles=numAcq+2;
	uint32_t i=0,idx;
	char fname[1024],nextFname[1024];
	int timeout;

	if(numThreads > 1)
		endIdx = (numDirs*(threadNum+1))/numThreads - 1;
	else
		endIdx = numDirs;


	if(verbose)
		printf("Thread %" PRIu64 ":  startIdx=%d endIdx=%d numFiles=%d numDirs=%d\n",threadNum,startIdx,endIdx,numFiles,numDirs);

	for(i=0;i<numFiles;i++)
	{
		for(idx=startIdx;idx<endIdx;idx++)
		{
			GetFileName(idx,i,fname,sizeof(fname));
			GetFileName(idx,i+1,nextFname,sizeof(nextFname));

			// wait for the file to exist if requested
			timeout=300;
			while((--timeout > 0)  && (stat(nextFname,&statBuf) != 0))
			{
				sleep(1); // wait for the next file to be available
			}

			if(timeout <= 0)
			{
				printf("timed out waiting for file %s\n",nextFname);
				Error = 1;
				exit(-1);
			}
			// check the current file
			ChkFile(fname);
		}
	}
	return NULL;
}

uint32_t GetNumAcq()
{
	// try opening explog.txt in cur dir.

	FILE *fp;
	uint32_t rc=0;
	char name[1024];
	char buf[4096];
	int frc;
	sprintf(name,"%s/explog.txt",DstDir);
	fp = fopen(name,"r");
	if(fp == NULL)
	{
		sprintf(name,"%s/../explog.txt",DstDir);
		fp = fopen(name,"r");
	}

	if(fp)
	{
		const char *label="Flows:";
		memset(buf,0,sizeof(buf));
		frc = fread(buf,1,sizeof(buf)-1,fp);
		if(frc > 0)
		{
			char *str = strstr(buf,label);
			if(str)
			{
				sscanf(str+strlen(label),"%d",&rc);
			}
		}
		fclose(fp);
	}
	return rc;
}


int main(int argc, char *argv[])
{
//	int cropx = 624, cropy = 125, cropw = 100, croph = 100;
	uint64_t i;

    size_t maxX = 0;
    size_t maxY = 0;
    size_t maxF = 0;

	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'n':
				argcc++;
				sscanf(argv[argcc],"%d",&numThreads);
				printf("NumThreads = %d\n",numThreads);
                break;

			case 'f':
				argcc++;
				oneFile = argv[argcc];
				printf("File to check = %s\n",oneFile);
                break;

			case 'r':
				argcc++;
				DstDir = argv[argcc];
				printf("Directory to check = %s\n",DstDir);
                break;

			case 'v':
				verbose=1;
                break;

            case 'w':
                dont_wait = 1;
                break;

            case 'X':
                argcc++;
                maxX = atoi(argv[argcc]);
                break;

            case 'Y':
                argcc++;
                maxY = atoi(argv[argcc]);
                break;

            case 'F':
                argcc++;
                maxF = atoi(argv[argcc]);
                break;

            case 'H':
			default:
				argcc++;
				fprintf (stdout, "\n");
				fprintf (stdout, "ChkDat - Utility to check the validity of dat files.\n");
				fprintf (stdout, "options:\n");
				fprintf (stdout, "   -n <numThr> \tNumber of threads\n");
				fprintf (stdout, "   -v\tverbose\n");
				fprintf (stdout, "   -f\tfile to check\n");
                fprintf (stdout, "   -F\tmax flow\n");
                fprintf (stdout, "   -X\tmax x (cols)\n");
                fprintf (stdout, "   -Y\tmax y (rows)\n");
                fprintf (stdout, "   -r\trecursive directory to check\n");
				fprintf (stdout, "   -w\tdon't wait for files\n");
				fprintf (stdout, "   -H\tPrints this message and exits.\n");
				fprintf (stdout, "\n");
				fprintf (stdout, "usage:\n");
				fprintf (stdout, "   ChkDat -r /results/analysis/PGM/testRun1\n");
				fprintf (stdout, "\n");
				exit (1);
		}
		argcc++;
	}

	if(oneFile)
	{
		// check the single file
        ChkFile(oneFile,maxX,maxY,maxF,verbose);
	}
	else
	{
		// create the list of directories
		pthread_t thr[numThreads];

		memset(&DirList,0,sizeof(DirList));
		// populate dirlist
		DIR *d = opendir(DstDir);
		struct dirent *entry;
		if(d)
		{
		    while ((entry = readdir(d)) != NULL)
		    {
		    	if ((entry->d_type == DT_DIR) && strcmp(entry->d_name,".") && strcmp(entry->d_name,".."))
		    	{
		    		strcpy(DirList[numDirs++],entry->d_name);
			    	printf(" Checking %s\n", entry->d_name);
		    	}
		    }
		    closedir(d);
		    if(numDirs==0)
		    {
		    	// didn't add anything..
		    	strcpy(DirList[numDirs++],".");
		    }
		}

		// fill in the number of acquisitions to check
		if(numAcq == 0)
			numAcq = GetNumAcq();

		if(numThreads > numDirs)
			numThreads = numDirs;

		// spawn the worker threads
		for(i=0;i<numThreads;i++)
		{
			pthread_create(&thr[i],NULL,worker,(void *)i);
		}

		// wait for them to complete
		for(i=0;i<numThreads;i++)
		{
			pthread_join(thr[i],NULL);
		}

	}


	printf("All OK\n\n");
}

