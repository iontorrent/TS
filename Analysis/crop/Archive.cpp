/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
//
// Archive - a tool used to continuously compress & archive experiment data as required
// in order to keep R&D/raw experiment repository storage area available
//

// Overview of approach:
// 1.  wake up and check drive space, once every 5 minutes, when more than 70% full, goto step 2
// 2.  gather list of all experiment directories, sort by access time
// 3.  if experiment is not compressed, compress, else rsync off to backup drive
// 4.  check drive space, goto 3 as long as there are still experiments to process & space exceeds 70% full

// note: using the access time allows us to first compress an experiment but keep it around,
// then archive it off most likely in its compressed form.  Converting to compressed will update
// the access time, wo this file will not be touched for a while, but will eventually be archived

// if a user continually accesses an experiment or compressed experiment, that experiment will remain
// on the drive indefinitely.  A nice feature from the user's perspective

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
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
#include <assert.h>
#include "ByteSwapUtils.h"
#include "datahdr.h"
#include "LinuxCompat.h"
// #include "Raw2Wells.h"
#include "Image.h"
#include "Acq.h"
#include "Utils.h"

char backupDir[MAX_PATH_LENGTH] = {0};
bool ok = false;
char *bkupExt = "Bkup_";

struct ExpList {
	time_t	atime;
	char	*instName;
	char	*expName;
	bool	compressed;
};

struct HDR {
	_expmt_hdr_v3	expHdr;
};

void GetExperimentList(char *dirToMonitor, ExpList **expList, int *numExperiments);
void Archive(char *dirToMonitor, ExpList e);
void Compress(char *dirToMonitor, ExpList e);
double GetFreeSpace(char *rootDir);
void GetBackupDrive();
bool IsCompressed(char *dirToMonitor, char *instrumentName, char *experimentName);

int main(int argc, char *argv[])
{
	char	*dirToMonitor = "/results";
	double	freeSpace = 0.0;
	double	spaceThresh = 0.3; // below this amount of free space and we spring into action
	bool	allowCompress = false;
	bool	allowBackup = false;

	int argcc = 1;
	while (argcc < argc) {
		switch (argv[argcc][1]) {
			case 'c': // allow compression to occur
				allowCompress = true;
			break;

			case 'b': // allow backups to occur
				allowBackup = true;
			break;

			case 'd': // results directory to monitor
				argcc++;
				dirToMonitor = argv[argcc];
			break;

			case 'e': // backup extension to use
				argcc++;
				bkupExt = argv[argcc];
			break;
		}
		argcc++;
	}

	printf("Monitoring: %s\n", dirToMonitor);
	if (allowCompress)
		printf("Compression allowed.\n");
	if (allowBackup)
		printf("Backup allowed to %s*\n", bkupExt);
	ok = true;

	do {
		// step 1
		freeSpace = GetFreeSpace(dirToMonitor);
		GetBackupDrive();

		if (freeSpace < spaceThresh) {
			printf("Drive space low (%.2lf%% free)\n", freeSpace*100.0);
			// step 2
			ExpList *experimentList = NULL;
			int numExperiments = 0;
			GetExperimentList(dirToMonitor, &experimentList, &numExperiments);
			printf("Found %d experiments\n", numExperiments);
			int i;
			for(i=0;i<numExperiments;i++) {
				// step 3
				if (allowCompress && !IsCompressed(dirToMonitor, experimentList[i].instName, experimentList[i].expName)) {
					Compress(dirToMonitor, experimentList[i]);
				} else {
					if (allowBackup)
						Archive(dirToMonitor, experimentList[i]);
				}

				// step 4
				freeSpace = GetFreeSpace(dirToMonitor);
				if (freeSpace >= spaceThresh)
					break;
			}
			for(i=0;i<numExperiments;i++) {
				free(experimentList[i].expName);
				free(experimentList[i].instName);
			}
			if (numExperiments > 0)
				free(experimentList);
		}

		printf("Sleeping...\n");
		sleep(60*5); // go to sleep for 5 minutes
		// sleep(5); // go to sleep for 5 minutes
	} while (ok);
}

bool IsCompressed(char *dirToMonitor, char *instrumentName, char *experimentName)
{
	bool compressed = false;

	char acqName[MAX_PATH_LENGTH];
	sprintf(acqName, "%s/%s/%s/acq_0000.dat", dirToMonitor, instrumentName, experimentName);
	FILE *fp = fopen(acqName, "rb");
	if (fp) {
		HDR     hdr;
		int elements_read = fread(&hdr, sizeof(HDR), 1, fp);
        assert(elements_read == 1);
		fclose(fp);
		ByteSwap2(hdr.expHdr.rows);
		ByteSwap2(hdr.expHdr.cols);
		ByteSwap2(hdr.expHdr.channels);
		ByteSwap2(hdr.expHdr.interlaceType);
		if ((hdr.expHdr.channels == 1) && (hdr.expHdr.interlaceType == 0) && (hdr.expHdr.rows < 1000) && (hdr.expHdr.cols < 1000))
			compressed = true;
	}

	return compressed;
}

void AddExperiment(char *dirToMonitor, ExpList **expList, int *num, char *instrumentName, char *experimentName)
{
	// get last access time
	char pathAndName[MAX_PATH_LENGTH];
	sprintf(pathAndName, "%s/%s/%s", dirToMonitor, instrumentName, experimentName);
	struct stat fileInfo;
	int fd = open(pathAndName, O_RDONLY);
	if (fstat(fd, &fileInfo) == 0) {
		// add new experiment node
		int i = *num;
		*expList = (ExpList *)realloc(*expList, sizeof(ExpList) * (i+1));
		*num = *num + 1;
		(*expList)[i].atime = fileInfo.st_mtime;
		(*expList)[i].instName = strdup(instrumentName);
		(*expList)[i].expName = strdup(experimentName);
		(*expList)[i].compressed = false;
// printf("Added %s/%s %d with atime: %lu\n", (*expList)[i].instName, (*expList)[i].expName, i, (*expList)[i].atime);
		close(fd);
	} else {
        close(fd);
    }
}

int ExpListCompare(const void *_item1, const void *_item2)
{
	ExpList *item1 = (ExpList *)_item1;
	ExpList *item2 = (ExpList *)_item2;
	return (item1->atime - item2->atime);
}

void SortExperiment(ExpList *expList, int num)
{
	if (num > 1)
		qsort(expList, num, sizeof(ExpList), ExpListCompare);
}

void GetExperimentList(char *dirToMonitor, ExpList **expList, int *numExperiments)
{
	// expreiments are assumed to have a general directory format as follows:
	// dirToMonitor/INSTRUMENT/EXPERIMENT
	// so we look for all directories under the dirToMonitor, and for each, find matching dirs with 'R_'
	printf("Generating experiment list...\n");
	*numExperiments = 0;
	*expList = NULL;

	DIR *dir;
	struct dirent *dp;
	if ((dir = opendir(dirToMonitor)) != NULL) {
		while ((dp = readdir(dir)) != NULL) {
			if (dp->d_type == DT_DIR && strncmp(dp->d_name, ".", 1)) {
				DIR *instrumentDir;
				struct dirent *experiment;
				char path[MAX_PATH_LENGTH];
				sprintf(path, "%s/%s", dirToMonitor, dp->d_name);
				if ((instrumentDir = opendir(path)) != NULL) {
					printf("Reading from instrument: %s\n", dp->d_name);
					while((experiment = readdir(instrumentDir)) != NULL) {
						if (experiment->d_type == DT_DIR && strncmp(experiment->d_name, "R_", 2) == 0) {
							AddExperiment(dirToMonitor, expList, numExperiments, dp->d_name, experiment->d_name);
						}
					}
					closedir(instrumentDir);
				}
			}
		}
		closedir(dir);
	}

	SortExperiment(*expList, *numExperiments);
}

void Compress(char *dirToMonitor, ExpList e)
{
	printf("Compressing experiment: %s/%s/%s\n", dirToMonitor, e.instName, e.expName);

	int x, y, w, h;
	// set region to "midchip"
	x = 524;
	y = 125;
	w = 300;
	h = 300;

	Image loader;
	Acq saver;
	int i;
	char name[MAX_PATH_LENGTH];
	for(i=0;i<500;i++) {
		sprintf(name, "%s/%s/%s/acq_%04d.dat", dirToMonitor, e.instName, e.expName, i);
		if (loader.LoadRaw(name, 0, (i==0 ? true : false), false)) {
			saver.SetData(&loader);
			if (!saver.Write(name, x, y, w, h))
				break;
			printf("Converted: %s\n", name);
		}
	}
	loader.Cleanup(); // will allow us to re-allocate mem since the beadfind data might have been imaged for longer

	// now do any pre/post beadfinds and prerun data
	char *nameList[] = {"beadfind_post_0000.dat", "beadfind_post_0001.dat", "beadfind_post_0002.dat", "beadfind_post_0003.dat",
			"beadfind_pre_0000.dat", "beadfind_pre_0001.dat", "beadfind_pre_0002.dat", "beadfind_pre_0003.dat",
			"prerun_0000.dat", "prerun_0001.dat", "prerun_0002.dat", "prerun_0003.dat", "prerun_0004.dat"};
	for(i=0;i<13;i++) {
		sprintf(name, "%s/%s/%s/%s", dirToMonitor, e.instName, e.expName, nameList[i]);
		if (loader.LoadRaw(name, 0, (i==0 ? true : false), false)) {
			saver.SetData(&loader);
			if (!saver.Write(name, x, y, w, h))
				break;
			printf("Converted: %s\n", name);
		}
	}

	loader.Cleanup(); // will allow us to re-allocate mem since the beadfind data might have been imaged for longer
	saver.SetData(NULL);
}

void Archive(char *dirToMonitor, ExpList e)
{
	if (backupDir[0] == 0) {
		printf("Warning, no drive present to backup to, aborting!\n");
		return;
	}

	printf("Archiving experiment: %s/%s/%s\n", dirToMonitor, e.instName, e.expName);

	char cmd[512];
	// first make sure directories exist, make them if not
	sprintf(cmd, "mkdir /media/%s/%s/%s", backupDir, dirToMonitor, e.instName);
	int rc = system(cmd);
    assert(rc==0);
	sprintf(cmd, "mkdir /media/%s/%s/%s/%s", backupDir, dirToMonitor, e.instName, e.expName);
	rc = system(cmd);
    assert(rc==0);

	// now backup (rsync) the files
	sprintf(cmd, "rsync -pogt --progress --relative --bwlimit=40000 -r %s/%s/%s /media/%s", dirToMonitor, e.instName, e.expName, backupDir);
	if (system(cmd) == 0) {
		// rsync completed successfully, delete dir
		sprintf(cmd, "rm -rf %s/%s/%s", dirToMonitor, e.instName, e.expName);
		rc = system(cmd);
        assert(rc==0);
		// make symlink to backup drive instead
		sprintf(cmd, "ln -s /media/%s/%s/%s/%s %s/%s/%s", backupDir, dirToMonitor, e.instName, e.expName, dirToMonitor, e.instName, e.expName);
		rc = system(cmd);
        assert(rc==0);
	} else {
		printf("rsync failed on %s/%s/%s\n", dirToMonitor, e.instName, e.expName);
		backupDir[0] = 0; // lets not try and sync up more data to this drive!
	}
}

double GetFreeSpace(char *rootDir)
{
	struct statfs buf;
	memset(&buf, 0, sizeof(buf));
	double percentFree = 0.0;
	if (statfs(rootDir, &buf) == 0) {
		if (buf.f_blocks > 0)
			percentFree = (double)buf.f_bavail/(double)buf.f_blocks;
	}

	printf("Drive %s free percent: %.2lf%%\n", rootDir, percentFree*100.0);
	return percentFree;
}

void GetBackupDrive()
{
	backupDir[0] = 0;
	DIR *dir;
	struct dirent *dp;
	if ((dir = opendir("/media")) != NULL) {
		while ((dp = readdir(dir)) != NULL) {
			if (dp->d_type == DT_DIR && (strncmp(dp->d_name, "Bkup_", 5) == 0)) {
				if ((backupDir[0] == 0) || (strcmp(backupDir, dp->d_name) < 0))
					strcpy(backupDir, dp->d_name);
			}
		}
		closedir(dir);
	}

	if (backupDir[0] != 0) {
		char name[MAX_PATH_LENGTH];
		sprintf(name, "/media/%s", backupDir);
		double freeSpace = GetFreeSpace(name);
		printf("Found backup drive %s with %.2lf%% free space\n", backupDir, 100.0*freeSpace);
		if (freeSpace < 0.05) {
			backupDir[0] = 0;
			printf("Space too low on backup, not using.\n");
		}
	} else {
		printf("No reasonable backup drive found?\n");
	}
}

