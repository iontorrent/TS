/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#include "Utils.h"
#ifndef ALIGNSTATS_IGNORE
#include "IonVersion.h"
#endif
#include <sys/stat.h>
#include <libgen.h>
#include <limits.h>
#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream> // for istringstream

using namespace std;

char *readline(FILE *fp)
{
    char* line = (char*) calloc(1, sizeof(char));
    char c;
    int len = 0;
    while ((c = fgetc(fp)) != EOF && c != '\n') {
        line = (char *) realloc (line, sizeof(char) * (len + 2));
        line[len++] = c;
        line[len] = '\0';
    }
    return (line);
}

#ifndef ALIGNSTATS_IGNORE
char *GetExpLogParameter (const char *filename, const char *paramName)
{
    FILE *fp = NULL;
    fopen_s (&fp, filename, "rb");
    if (!fp) {
        strerror (errno);
        return (NULL);
    }
    char *parameter = NULL;
    size_t size = 256;	//getline resets line size as needed
	char *line = (char *)malloc (size);
	while ((getline(&line,&size,fp)) > 0) {
		//fprintf (stderr, "Size is %d\n", size);
		if (strstr (line, paramName)) {
            char *sPtr = strchr (line, ':');
            sPtr++; //skip colon
            //skip leading white space
			while (isspace(*sPtr)) sPtr++;
            parameter = (char *) malloc (sizeof(char) * (size + 2));
            strcpy (parameter, sPtr);
        }
    }
	if (line)
		free(line);
	fclose (fp);
	//DEBUG
	//fprintf (stdout, "getExpLogParameter: %s %s\n", paramName, parameter);
    return (parameter);
}

void GetExpLogParameters (const char *filename, const char *paramName, std::vector<std::string> &values)
{
	FILE *fp = NULL;
	values.resize(0);
	fopen_s (&fp, filename, "rb");
	if (!fp) {
		strerror (errno);
		return;
	}
	size_t size = 256;	//getline resets line size as needed
	char *line = (char *)malloc (size);
	while ((getline(&line,&size,fp)) > 0) {
		//fprintf (stderr, "Size is %d\n", size);
		if (strstr (line, paramName)) {
			char *sPtr = strchr (line, ':');
            sPtr++; //skip colon
            //skip leading white space
			while (isspace(*sPtr)) sPtr++;
			values.push_back(sPtr);
		}
	}
	if (line) {
		free(line);
	}
	fclose (fp);
}
#endif


//
//  Tests a string for containing a valid filesystem path
//
bool isDir (const char *path)
{
    struct stat x;
    if (stat(path, &x) != 0)
        return false;
    return (S_ISDIR(x.st_mode) ? true:false);
}
//
//  Tests a string for containing a valid filesystem file
//
bool isFile (const char *path)
{
    struct stat x;
    if (stat(path, &x) != 0)
        return false;
    return (S_ISREG(x.st_mode) ? true:false);
}

#ifndef ALIGNSTATS_IGNORE
//
//	Parses explog.txt and returns a flag indicating whether wash flows are
//	present in the run data.
//	Returns 0 - no wash flow
//			1 - wash flow present
//		   -1 - could not determine
//
int HasWashFlow (char *datapath)
{
	int washImages = -1;	// Set default to indicate error determine value.
	char *filepath = NULL;
	char *argument = NULL;
	filepath = getExpLogPath(datapath);
	
	argument = GetExpLogParameter (filepath,"Image Map");
	if (argument) {
		//	Typical arg is 
		//	"5 0 r4 1 r3 2 r2 3 r1 4 w2"	- would be wash flow
		// 	"4 0 r4 1 r3 2 r2 3 r1"	- would be no wash
		int flowNum;
		
		sscanf (argument,"%d", &flowNum);
		if (flowNum == 5) {
			char *sPtr = strrchr (argument, 'w');
            if (sPtr)
                washImages = 1;
            else
                washImages = 0;
		}
		else if (flowNum == 4) {
			washImages = 0;
		}
		else {
			// Its not either the expected 5 or 4.
            //Check the last part of the string for a 'w' character
            char *sPtr = strrchr (argument, 'w');
            if (sPtr)
                washImages = 1;
            else
                washImages = 0;
		}
		free (argument);
	}
	
	if (filepath) free (filepath);
	
	return (washImages);
}

char	*GetPGMFlowOrder (char *path)
{
	char *flowOrder = NULL;
	char *filepath = NULL;
	char *argument = NULL;
	filepath = getExpLogPath(path);
	
	argument = GetExpLogParameter (filepath,"Image Map");
	const char mapping[6] = {"0GCAT"};
	if (argument) {
		
		//	Typical arg is 
		//	"5 0 r4 1 r3 2 r2 3 r1 4 w2"	- would be wash flow
		// 	"4 0 r4 1 r3 2 r2 3 r1"	- would be no wash
		//  -OR-
		//	"4 0 T 1 A 2 C 3 G"
		//	-OR-
		//	"tcagtcagtcag"
		//
		//fprintf (stderr, "Raw string = '%s'\n", argument);
		
		// First entry is number of flows in cycle, unless its not!
		int numFlows = 0;
		sscanf (argument,"%d", &numFlows);
		if (numFlows == 0) {
			numFlows = strlen(argument);
		}
		assert (numFlows > 0);
		
		//	allocate memory for the floworder string
		flowOrder = (char *) malloc (numFlows+1);
		
		
		//	Upper case everything
		ToUpper (argument);
		//fprintf (stdout, "argument = '%s'\n", argument);
		
		//  Read string char at a time.
		//		If its 'R' then
		//			read next char as an integer and convert integer to Nuke
		//		else if its T A C or G
		//			set Nuke
		//		else skip
		int num = 0;	//increments index into argument string
		int cnt = 0;	//increments index into flowOrder string
		for (num = 0; argument[num] != '\n'; num++){
			//fprintf (stdout, "We Got '%c'\n", argument[num]);
			if (argument[num] == 'R') {
				//this will work as long as there are less than 10 reagent bottles on the PGM
				int index = argument[num+1] - 48;	//does this make anyone nervous?
				//fprintf (stdout, "Index = %d\n", index);
				assert (index > 0 && index < 9);
				//fprintf (stdout, "mapping[%d] = %c\n", index, mapping[index]);
				flowOrder[cnt++] = mapping[index];
				flowOrder[cnt] = '\0';
			}
			else if (argument[num] == 'T' ||
					 argument[num] == 'A' ||
					 argument[num] == 'C' ||
					 argument[num] == 'G') {
				flowOrder[cnt++] = argument[num];
				flowOrder[cnt] = '\0';
			}
			else {
				//skip this character
			}
		}

		free (argument);
	}
	if (filepath) free (filepath);
	return (flowOrder);
}
#endif

//
//Converts initial portion of a string to a long integer, with error checking
//
bool validIn (char *inStr, long *value)
{
    char *endPtr = NULL;
    
    errno = 0;
    *value = strtol (inStr, &endPtr, 10);
    if ((errno == ERANGE && (*value == LONG_MAX || *value == LONG_MIN))
                   || (errno != 0 && *value == 0)) {
        perror("strtol");
        return (EXIT_FAILURE);
    }

    if (endPtr == inStr) {
        fprintf(stderr, "No digits were found\n");
        return (EXIT_FAILURE);
    }
    //fprintf (stdout, "Converted to %ld\n", *value);
    return EXIT_SUCCESS;
}

// determine if a string is a valid number (of any type; int, float, etc...)
bool isNumeric(char const* str, int numberBase)
{
	std::istringstream iss(str);

	if (numberBase == 10)
	{
		double doubleSink;
		iss >> doubleSink;
	}
	else if (numberBase == 8 || numberBase == 16)
	{
		int intSink;
		iss >> ((numberBase == 8) ? oct : hex ) >> intSink;
	}
	else
		return false;

	// was any input successfully consumed/converted?
	if (!iss)
		return false;

	// was all the input successfully consumed/converted?
	return (iss.rdbuf()->in_avail() == 0);
}

// convert a string to a floating point value
double ToDouble(char const* str)
{
	std::istringstream iss(str);
	double doubleSink;

	iss >> doubleSink;
	return doubleSink;
}

int numCores ()
{
#if 1
    // returns physical cpu cores
    int cores = 0;
    int processors = 0;
    int n = 0; //number elements read
    FILE *fp = NULL;
    // Number of cores
    fp = popen ("cat /proc/cpuinfo | grep \"cpu cores\" | uniq  | awk '{ print $4 }'", "r");

    // if the grep finds nothing, then this returns a NULL fp...
    if (fp == NULL)
        return(sysconf(_SC_NPROCESSORS_ONLN));

    n = fscanf (fp, "%d", &cores);
    if(n != 1)
      cores=1;
    pclose (fp);
    
    // Number of processors
    fp = popen ("cat /proc/cpuinfo | grep \"physical\\ id\" | sort | uniq | wc -l", "r");

    if (fp == NULL)
        return(sysconf(_SC_NPROCESSORS_ONLN));

    n = fscanf (fp, "%d", &processors);
    if(n != 1)
      processors=1;
    pclose (fp);
    
    /* Hack: Some VMs report 0 cores */
    cores = (cores > 0 ? cores:1);
    processors = (processors > 0 ? processors:1);
    
    return (cores * processors);
#else
    // returns virtual cpu cores
	return (sysconf(_SC_NPROCESSORS_ONLN));
#endif
}

//
//	Convert all chars in a string to upper case
//
void ToUpper (char *in)
{
	for (int i=0;in[i];i++)
		in[i] = toupper(in[i]);
}//
//	Convert all chars in a string to lower case
//
void ToLower (char *in)
{
	for (int i=0;in[i];i++)
		in[i] = tolower(in[i]);
}
//
//  Returns path of executable
//  Not portable - uses the proc filesystem
void get_exe_name(char * buffer)
{
	char linkname[64] = {0};
	pid_t pid;
	unsigned long offset = 0;

	pid = getpid();
	snprintf(&linkname[0], sizeof(linkname), "/proc/%i/exe", pid);

	if (readlink(&linkname[0], buffer, PATH_MAX) == -1)
		offset = 0;
	else
	{
		offset = strlen(buffer);
		while (offset && buffer[offset - 1] != '/') --offset;
		if (offset && buffer[offset - 1] == '/') --offset;
	}

	buffer[offset] = 0;
}
//
//  Return fully qualified path to some configuration files
//	Search order:
//		$HOME directory
//		$ION_CONFIG directory
//		Relative to executable's directory ../config/
//		Absolute path: /opt/ion/config/
//		Absolute path: /opt/ion/alignTools/
//
char *GetIonConfigFile (const char filename[])
{   
    char *string = NULL;
	
    //fprintf (stdout, "# DEBUG: Looking for '%s'\n", filename);
    // Search for TF config file:
    //  Current working directory
    size_t bufsize = 512;
    char buf[bufsize];
    assert(getcwd (buf, bufsize));    
    string = (char *) malloc (strlen (filename) + bufsize + 2);
    sprintf (string, "%s/%s", buf, filename);
    if (isFile (string)) {
        return (string);
    }
    else {
        free (string);
    }

    // Search for config file in $HOME:
    //  $HOME
    char *HOME = getenv("HOME");
    if (HOME) {
        string = (char *) malloc (strlen (filename) + strlen(HOME) + 2);
        sprintf (string, "%s/%s", HOME, filename);
        if (isFile (string)) {
			fprintf (stdout, "Found ... %s\n", string);
			return (string);
		}
		else {
			free (string);
		}
    }
    
    // Search for config file in $ION_CONFIG:
    //  Installation environment variable
    char *ION_CONFIG = getenv("ION_CONFIG");
    if (ION_CONFIG) {
        string = (char *) malloc (strlen (filename) + strlen(ION_CONFIG) + 2);
        sprintf (string, "%s/%s", ION_CONFIG, filename);
        if (isFile (string)) {
			return (string);
		}
		else {
			free (string);
		}
    }
    
    // Search for config file:
    //  Last ditch effort: Installation location.  Get location of binary, then up one dir and down into config
    char INSTALL[PATH_MAX] = {0};
    get_exe_name (INSTALL);
	// executable is always in bin so we specifically strip that off.
	char *sPtr = NULL;
	sPtr = strrchr (INSTALL, '/');
	if (sPtr)
		*sPtr = '\0';
    string = (char *) malloc (strlen (filename) + strlen(INSTALL) + strlen("config") + 3);
    sprintf (string, "%s/config/%s", INSTALL, filename);
    if (isFile (string)) {
		return (string);
    }
    else {
        free (string);
    }
	
	// Ultimate last ditch: hardcoded path
    string = (char *) malloc (strlen (filename) + strlen("/opt/ion/config") + 2);
	sprintf (string, "/opt/ion/config/%s", filename);
    if (isFile (string)) {
		return (string);
	}
	else {
		free (string);
	}
	
	// (YALDE): Yet another Ultimate last ditch: hardcoded path
    string = (char *) malloc (strlen (filename) + strlen("/opt/ion/alignTools") + 2);
	sprintf (string, "/opt/ion/alignTools/%s", filename);
    if (isFile (string)) {
		return (string);
	}
	else {
		free (string);
	}
	
	fprintf (stderr, "Cannot find Ion Config file: %s\n", filename);
	return (NULL);
}

//Copy a file
bool CopyFile(char *filefrom, char *fileto)
{
//#define printTime
#ifdef printTime
	time_t startTime;
	time_t endTime;
	time(&startTime);
#endif
    
	int size = 4096;
	char cmd[size];
	int alloc = snprintf (cmd, size-1, "cp %s %s",filefrom, fileto);
	//int alloc = snprintf (cmd, size-1, "cp %s %s && chmod a+rw %s &",filefrom, fileto, fileto);
	if (alloc < 0 || alloc > size-1) {
		fprintf (stderr, "CopyFile could not execute system copy command:\n");
		fprintf (stderr, "Copy file: %s\n", filefrom);
		fprintf (stderr, "To: %s\n", fileto);
		return (1);
	}

	int status;
	status = system(cmd);
	if (WIFEXITED(status)) {
		if (WEXITSTATUS(status)) {
			// error encountered
      fprintf(stderr, "From: %s\n", filefrom);
      fprintf(stderr, "To: %s\n", fileto);
      fprintf(stderr, "Command: %s\n", cmd);
			fprintf (stderr, "system copy command returned status %d\n", WEXITSTATUS(status));
			return (1);
		}
	}
    
    /* When 1.wells get copied, they have permissions of 600 and we like 666 */
    //snprintf (cmd, size-1, "chmod a+rw %s", fileto);
	/* Changing permissions to not allow others write*/
    snprintf (cmd, size-1, "chmod u=rw,g=rw,o=r %s", fileto);
    status = system(cmd);
	if (WIFEXITED(status)) {
		if (WEXITSTATUS(status)) {
			// error encountered
			fprintf (stderr, "chmod command returned status %d\n", WEXITSTATUS(status));
			return (1);
		}
	}
    

#ifdef printTime
	time(&endTime);
    struct stat buf;
    lstat(filefrom, &buf);
	fprintf (stdout, "Copy Time: %0.1lf sec. (%ld bytes)\n", difftime (endTime, startTime), (long int)buf.st_size);
	fprintf (stderr, "Copy file: %s\n", filefrom);
	fprintf (stderr, "To: %s\n", fileto);
#endif
    
	return 0;
}

/*
 *	Remove temporary 1.wells files leftover from previous Analysis
 */
void	ClearStaleWellsFile (void)
{
	DIR *dirfd;
	struct dirent *dirent;
	dirfd = opendir ("/tmp");
	while ((dirent = readdir(dirfd)) != NULL)
	{
		if (!(strstr(dirent->d_name, "well_")))
			continue;
		
		int pid;
		DIR *tmpdir = NULL;
		char name[MAX_PATH_LENGTH];
		char trash[MAX_PATH_LENGTH];
		sscanf (dirent->d_name,"well_%d_%s", &pid, trash);
		sprintf (name, "/proc/%d", pid);
		if ((tmpdir = opendir(name))) {
			closedir (tmpdir);
		}
		else {
			char creamEntry[MAX_PATH_LENGTH];
			sprintf (creamEntry, "/tmp/%s", dirent->d_name);
			unlink (creamEntry);
		}
	}
	closedir (dirfd);
}

/*
 *	Remove temporary 1.wells files leftover from previous Analysis
 */
void	ClearStaleSFFFiles (void)
{
	DIR *dirfd;
	struct dirent *dirent;
	const char *files[] = {"_rawlib.sff","_rawtf.sff"};
	for (int i = 0;i < 2; i++){
		dirfd = opendir ("/tmp");
		while ((dirent = readdir(dirfd)) != NULL)
		{
			if (!(strstr(dirent->d_name, files[i])))
				continue;
			
			int pid;
			DIR *tmpdir = NULL;
			char name[MAX_PATH_LENGTH];
			char trash[MAX_PATH_LENGTH];
			sscanf (dirent->d_name,"%d_%s", &pid, trash);
			sprintf (name, "/proc/%d", pid);
			if ((tmpdir = opendir(name))) {
				closedir (tmpdir);
			}
			else {
				char creamEntry[MAX_PATH_LENGTH];
				sprintf (creamEntry, "/tmp/%s", dirent->d_name);
				unlink (creamEntry);
			}
		}
		closedir (dirfd);
	}
}

#ifndef ALIGNSTATS_IGNORE
//
//find number of cycles in dataset
//
int GetCycles (char *dir)
{
    int cycles = 0;

    // Method using the explog.txt
	char *filepath = NULL;
	char *argument = NULL;
    long value;
	filepath = getExpLogPath(dir);
	
	argument = GetExpLogParameter (filepath,"Cycles");
    if (argument) {
        if (validIn (argument, &value)) {
            fprintf (stderr, "Error getting num cycles from explog.txt\n");
            exit (1);
        }
        else {
            cycles = (int) value;
        }
		free (argument);
    }
	else {
		//DEBUG
		fprintf (stderr, "No Cycles keyword found\n");
	}
    free (filepath);
    return (cycles);
}

//
//Determine number of Flows in run from explog.txt
//
int GetTotalFlows (char *dir)
{
    int numFlows = 0;

    // Method using the explog.txt
	char *filepath = NULL;
	char *argument = NULL;
    long value;
	filepath = getExpLogPath(dir);
	
	argument = GetExpLogParameter (filepath,"Flows");
    if (argument) {
        if (validIn (argument, &value)) {
            fprintf (stderr, "Error getting num flows from explog.txt\n");
            exit (1);
        }
        else {
			//DEBUG
			//fprintf (stderr, "GetTotalFlows: '%s' '%d'\n", argument, (int) value);
            numFlows = (int) value;
        }
		free (argument);
    }
    else {
    	// No Flows keyword found - legacy file format pre Jan 2011
		//DEBUG
		fprintf (stderr, "No Flows keyword found\n");
        int cycles = GetCycles(dir);
        numFlows = 4 * cycles;
    }
    if (filepath) free (filepath);
	//fprintf (stderr, "Returning numFlows = %d\n",numFlows);
    return (numFlows);
}

//
//return chip id string
//
char * GetChipId (const char *dir)
{
	// Method using the explog.txt
	char *argument = NULL;
	char *filepath = NULL;
	filepath = getExpLogPath(dir);
	
	argument = GetExpLogParameter (filepath,"ChipType");
    if (argument) {
		char *chip = (char *) malloc (10);
		int len = strlen(argument);
		int y = 0;
		for (int i = 0; i<len;i++) {
			if (isdigit (argument[i]))
				chip[y++] = argument[i];
		}
		chip[y] = '\0';
    	free (filepath);
		free (argument);
    	return (chip);
    }
    if (filepath) free (filepath);
    return (NULL);
}
#endif

void GetChipDim(const char *type, int dims[2])
{
  if (type != NULL) {
    if (strncmp ("314",type,3) == 0) {
      dims[0] = 1280;
      dims[1] = 1152;
    } else if (strncmp ("324",type,3) == 0) {
      dims[0] = 1280;
      dims[1] = 1152;
    } else if (strncmp ("316",type,3) == 0) {
      dims[0] = 2736;
      dims[1] = 2640;
    } else if (strncmp ("318",type,3) == 0) {
      dims[0] = 3392;
      dims[1] = 3792;
    } else {
      dims[0] = 0;
      dims[1] = 0;
    }
  } else {
    dims[0] = 0;
    dims[1] = 0;
  }
}

int	GetNumLines (char *filename)
{
	int cnt = 0;
	FILE *fp = fopen (filename, "rb");
	if (!fp){
		perror(filename);
		return(-1);
	}
	while (!feof(fp)){
		if (fgetc(fp) == '\n')
			cnt++;
	}
	fclose (fp);
	return(cnt);
}

void Trim(char *buf)
{
	int len = strlen(buf);
	while (len > 0 && (buf[len-1] == '\r' || buf[len-1] == '\n'))
		len--;
	buf[len] = 0;
}
//
//	Opens processParameters.txt file and reads the argument for the given
//	keyword
char * GetProcessParam (const char *filePath, const char *pattern)
{
	FILE *fp = NULL;
	char *fileName = NULL;
	char *arg = NULL;
	char *keyword = NULL;
	char *argument = NULL;
	char buf[16384];
	char *sPtr = NULL;
	
	fileName = (char *) malloc (strlen (filePath)+strlen ("/processParameters.txt")+1);
	sprintf (fileName, "%s/%s", filePath, "processParameters.txt");
	
	fp = fopen (fileName, "rb");
	if (!fp) {
		perror (fileName);
		free (fileName);
		return (NULL);
	}
	
	free (fileName);
	
	while (fgets(buf, sizeof(buf), fp)) {
		Trim (buf);
		if ((sPtr = strchr (buf, '='))) {
			//allocate plenty of space for each component of the entry. and initialize
			keyword		= (char *) calloc (1,strlen(buf));
			argument	= (char *) calloc (1,strlen(buf));
			
			//separate the components at the '=' char, remove whitespace
			char *aPtr = sPtr+1;
			while (isspace(*aPtr)) aPtr++;
			strncpy (argument, aPtr, strlen(buf)-1);
			char *end = aPtr + strlen (aPtr) - 1;
			while(end > aPtr && isspace(*end)) end--;

			*sPtr = '\0';
			strncpy (keyword, buf, strlen (buf)-1);
			end = keyword + strlen (keyword) - 1;
			while(end > keyword && isspace(*end)) end--;
			
			//select the desired keyword.  note: whitespace would be a problem
			//if we searched for exact match.
			if (strstr (keyword, pattern)) {
				arg = (char *) malloc (strlen(argument)+1);
				strcpy (arg, argument);
				free (keyword);
				free (argument);
				break;
			}
			free (keyword);
			free (argument);
		}
	}

    fclose(fp);
	
	return (arg);
}
/*
 *	For given width and height chip, and input region index and number of regions, return
 *  Region structure for unique region specified.
 */
void	defineSubRegion (int rows, int cols, int runIndex, int regionIdx, Region *cropRegions)
{

	// regionIdx is number of regions to create.
	// runIndex is which region to return.
	switch (regionIdx)
	{
		case 4:
		{
			int xinc = cols/2;
			int yinc = rows/2;
			Region regions[regionIdx];
			int i;
			int x;
			int y;
			for(i = 0, x=0;x<cols;x+=xinc) {
				for(y=0;y<rows;y+=yinc) {
					regions[i].col = x;
					regions[i].row = y;
					regions[i].w = xinc;
					regions[i].h = yinc;
					if (regions[i].col + regions[i].w > cols) // technically I don't think these ever hit since I'm truncating to calc xinc * yinc
						regions[i].w = cols - regions[i].col; // but better to be safe!
					if (regions[i].row + regions[i].h > rows)
						regions[i].h = rows - regions[i].row;
					i++;
				}
			}
			
			cropRegions->col = regions[runIndex-1].col;
			cropRegions->row = regions[runIndex-1].row;
			cropRegions->w = regions[runIndex-1].w;
			cropRegions->h = regions[runIndex-1].h;
			
			break;
		}
		case 9:
		//break;
		case 16:
		//break;
		default:
			fprintf (stderr, "Unsupported region divisor: %d\n", regionIdx);
		break;
	}
	
}

bool IsValid(const double *vals, int numVals)
{
	int i;
	for(i=0;i<numVals;i++) {
		if (isnan(vals[i]))
			return false;
	}
	return true;
}

void FillInDirName(const string &path, string &dir, string &file) {
  size_t slashPos = path.rfind('/');
  if (slashPos != string::npos) {
    dir = path.substr(0, slashPos);
    file = path.substr(slashPos + 1, (path.length() - (slashPos+1)));
  }
  else {
    dir = ".";
    file = path;
  }
}

#ifndef ALIGNSTATS_IGNORE
void init_salute()
{
	char banner[256];
	sprintf (banner, "/usr/bin/figlet -f script Analysis %s 2>/dev/null", IonVersion::GetVersion().c_str());
	if (system(banner))
	{
		// figlet did not execute;
		fprintf (stdout, "%s\n", IonVersion::GetVersion().c_str());
	}
}
#endif

//
// updates a file called progress.txt which would be in the CWD when Analysis
// is launched by web interface.
//
bool updateProgress (int transition)
{
	char cmd[1024];
	char *path = NULL;
	bool error = false;
	bool debugme = false;
	
	path = strdup ("progress.txt");
	
	switch (transition)
	{
		// Transition from beadfind to image processing
		case WELL_TO_IMAGE:
		
			if (debugme) fprintf (stderr, "Changing color of wellproc\n");
			sprintf (cmd,
					 "sed -i 's/wellfinding.*/wellfinding = green/' "
					 "%s 2>/dev/null", path);
			if (system (cmd)) {
				error = true;
				if (debugme) fprintf (stderr, "Error running system cmd\n");
			}
				
			if (debugme) fprintf (stderr, "Changing color of signalproc\n");
			sprintf (cmd,
					 "sed -i 's/signalprocessing.*/signalprocessing = yellow/' "
					 "%s 2>/dev/null", path);
			if (system (cmd)) {
				error = true;
				if (debugme) fprintf (stderr, "Error running system cmd\n");
			}
		break;
		// Transition from image processing to signal processing
		case IMAGE_TO_SIGNAL:
		
			if (debugme) fprintf (stderr, "Changing color of signalproc\n");
			sprintf (cmd,
					 "sed -i 's/signalprocessing.*/signalprocessing = green/' "
					 "%s 2>/dev/null", path);
			if (system (cmd)) {
				error = true;
				if (debugme) fprintf (stderr, "Error running system cmd\n");
			}
				
			if (debugme) fprintf (stderr, "Changing color of basecalling\n");
			sprintf (cmd,
					 "sed -i 's/basecalling.*/basecalling = yellow/' "
					 "%s 2>/dev/null", path);
			if (system (cmd)) {
				error = true;
				if (debugme) fprintf (stderr, "Error running system cmd\n");
			}
			
		break;
		default:
			fprintf (stderr, "Unknown transition: %d\n", transition);
			error = true;
		break;
	}

	free (path);
	return (error);
}

int	count_char(std::string s, char c) {
	
	size_t pos = 0;
	int tot = 0;
	string tmp = s;
	while (pos!=string::npos) {
		tmp = tmp.substr(pos+1);
		
		pos = tmp.find(c);
		if(pos != string::npos) {
			tot++;
		} 
		
	}
	if (tot > 0) {
		return tot;
	} else {
		return 0;
	}
	
	
	
}


string get_file_extension(const string& s) {
	
	size_t i = s.rfind('.', s.length( ));
	if (i != string::npos) {
		return(s.substr(i+1, s.length( ) - i));
	}
	
	return("");
}


void split(const string& s, char c, vector<string>& v) {
	string::size_type i = 0;
	string::size_type j = s.find(c);
	while (j != string::npos) {
		v.push_back(s.substr(i, j-i));
		i = ++j;
		j = s.find(c,j);
		
		if (j == string::npos) {
			v.push_back(s.substr(i, s.length()));
		}
	}
	
}

void uintVectorToString(vector<unsigned int> &v, string &s, string &nullStr, char delim) {
  if(v.size() > 0) {
    std::stringstream converter0;
    converter0 << v[0];
    s = converter0.str();
    for(unsigned int i=1; i<v.size(); i++) {
      std::stringstream converter1;
      converter1 << v[i];
      s += delim + converter1.str();
    }
  } else {
    s = nullStr;
  }
}

int seqToFlow(const char *seq, int seqLen, int *ionogram, int ionogramLen, char *flowOrder, int flowOrderLen)
{
	int flows = 0;
	int bases = 0;
	while (flows < ionogramLen && bases < seqLen) {
		ionogram[flows] = 0;
		while ((bases < seqLen) && (flowOrder[flows%flowOrderLen] == seq[bases])) {
			ionogram[flows]++;
			bases++;
		}
		flows++;
	}
	return flows;
}

void flowToSeq(string &seq, hpLen_vec_t &hpLen, string &flowOrder) {
  unsigned int cycleLen = flowOrder.size();
  seq.clear();
  if(cycleLen > 0) {
    for(unsigned int iFlow=0; iFlow < hpLen.size(); iFlow++) {
      char thisNuc = flowOrder[iFlow % cycleLen];
      for(char iNuc=0; iNuc < hpLen[iFlow]; iNuc++) {
        seq += thisNuc;
      }
    }
  }
}

//
// Returns pointer to string containing path to explog.txt file
// Can be in given raw data directory, or parent of given directory if its a gridded dataset
//
char * getExpLogPath (const char *dir)
{
	//first try the given directory - default behavior for monogrid data
	char filename[] = {"explog.txt"};
	char *filepath = NULL;
	filepath = (char *) malloc (sizeof(char) * (strlen (filename) + strlen (dir) + 2));
	sprintf (filepath, "%s/%s", dir, filename);
	if (isFile(filepath)) {
		return filepath;
	}
	free(filepath);
	//second try the parent directory
	char *parent = NULL;
	parent = strdup (dir);
	char *parent2 = dirname(parent);
	filepath = (char *) malloc (sizeof(char) * (strlen (filename) + strlen (parent2) + 2));
	sprintf (filepath, "%s/%s", parent2, filename);
	free(parent);
	if (isFile(filepath)) {
		return filepath;
	}
	return NULL;
}

void ChopLine(std::vector<std::string> &words, const std::string &line, char delim) {
  size_t current = 0;
  size_t next = 0;
  words.clear();
  while(current < line.length()) {
    next = line.find(delim, current);
    if (next == string::npos) {
      next = line.length();
    }
    words.push_back(line.substr(current, next-current));
    current = next + 1;
  }
}

std::string GetMemUsage() {
  pid_t pid =  getpid();
  string name = "/proc/" + ToStr(pid) + "/statm";
  std::ifstream file;
  file.open(name.c_str() , ifstream::in);
  string line;
  string usage;
  vector<string> words;
  if (getline(file, line)) {
    ChopLine(words, line, ' ');
  }
  if (words.size() < 3) {
    usage = "unknown";
  }
  else {
    size_t virt = atoi(words[0].c_str()) * 4 * 1024 / 1048576;
    size_t resident = atoi(words[1].c_str()) * 4 * 1024 / 1048576;                  
    usage = "Virtual: " + ToStr(virt) + "MB Resident: " + ToStr(resident) + "MB";
  }
  file.close();
  return usage;
}


void MemoryUsage(const std::string &s) {
  std::cout << "MEM USAGE: " << s << " - " << GetMemUsage() << std::endl;
}

void MemUsage(const std::string &s) { 
  MemoryUsage(s); 
}

void TrimString(std::string &str) {
  std::string whitespaces (" \t\f\v\n\r");
  size_t found = str.find_last_not_of(whitespaces);
  if (found != std::string::npos)
    str.erase(found+1);
  else
    str.clear(); 
  found = str.find_first_not_of(whitespaces);
  if (found != std::string::npos)
    str.erase(0,found);
  else
    str.clear(); 
}
int totalMemOnTorrentServer()
{
    const int totalMem = 48*1024*1024; // defaults to T7500
    FILE *fp = NULL;
    int mem;
    fp = popen ("cat /proc/meminfo | grep \"MemTotal:\" | awk '{ print $2 }'", "r");

    // if the grep finds nothing, then this returns a NULL fp...
    if (fp == NULL)
        return totalMem;

    int n = fscanf (fp, "%d", &mem);
    if (n != 1)
        mem = totalMem;

    pclose (fp);
    
    return mem;
}


