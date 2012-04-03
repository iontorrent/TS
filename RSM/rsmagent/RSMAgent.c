//
// Remote Support and Monitor Agent - Torrent Server
// (c) 2011 Life Technologies, Ion Torrent
//


#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/vfs.h>

#include <openssl/opensslv.h>

#include "AeOSLocal.h"
#include "AeTypes.h"
#include "AeError.h"
#include "AeOS.h"
#include "AeInterface.h"

#define DEFAULT_OWNER "drm-data_source"
#define TS_VERSIONS "./TSConfig.txt"
#define TS_SERVERS "./TSServers.txt"
#define ALT_SN "./serial_number.alt"
#define LOC_FILE "./loc.txt"

typedef enum {
	STATUS_HD = 0,
	STATUS_EVENT,
	STATUS_TASK,
	STATUS_VERSIONS,
	STATUS_SERVICES,
	STATUS_FILESERVERS,
	STATUS_PGMS,
	STATUS_CONTACTINFO,
	STATUS_NETWORK,
	STATUS_EXPERIMENT,
} StatusType;

typedef struct {
	StatusType	type;
	time_t		updateRate;
	time_t		lastUpdateTime;
} UpdateItem;

typedef struct {
	char	*modelNumber;
	char	*serialNumber;
	int	pingRate;
	int	numPGMs;
	char	**pgmSerialNumberList;
} AgentInfo;

typedef struct _AeDemoUpload AeDemoUpload;

struct _AeDemoUpload
{
    AeFileUploadSpec **ppUploads;
    AeInt32          iUploadIdx;
    AeFileStat       curFileStat;
    AeFileHandle     iCurFileHandle;
    AeChar           pBuffer[BUFSIZ];
};

// -- File system list management --
typedef struct {
	char	mountedName[256];
	char	agentAttributeName[256]; // just the last portion of the mounted name without the trailing slash
	double	percentFull;
} FileSystemList;
int numFileSystems = 0;
FileSystemList *fileSystemList = NULL;


// globals
AgentInfo agentInfo;
int ok = 1;
int verbose = 0;
unsigned long pingRate = 30;
time_t curTime, timeNow;
AeInt32 iDeviceId, iServerId;
static int nextPort = 15000;
static double percentFull = -1.0;
static time_t lastVersionModTime = 0;
static time_t lastContactInfoTime = 0;
static int wantConfigUpdate = 1;

UpdateItem updateItem[] = {
	{STATUS_VERSIONS, 120, 0},
	{STATUS_CONTACTINFO, 120, 0},
	{STATUS_EXPERIMENT, 120, 0},
	{STATUS_SERVICES, 360, 0},
	{STATUS_FILESERVERS, 600, 0},
	{STATUS_PGMS, 3600, 0},
	//{STATUS_NETWORK, 3600, 0},
};
int numUpdateItems = sizeof(updateItem) / sizeof(UpdateItem);

int UpdateDataItem(StatusType status, AeDRMDataItem *dataItem);
void SendVersionInfo(AeDRMDataItem *dataItem);
void SendServersStatus(AeDRMDataItem *dataItem);
void GenerateVersionInfo();
void SendFileServerStatus(AeDRMDataItem *dataItem);

// file upload callbacks
static AeBool OnFileUploadBegin(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AePointer *ppUserData);
static AeBool OnFileUploadData(AeInt32 iDeviceId, AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData);
static void OnFileUploadEnd(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData);

// GetConfigEntry - returns the requested entry into buf and returns 0 or error code
int GetConfigEntry(char *configFile, char delimiter, char *entry, char *buf, int bufSize)
{
	int ret = 1; // entry not found by default

	FILE *fp = fopen(configFile, "r");
	if (fp) {
		char line[1024];
		char *ptr;
		while (ret == 1 && fgets(line, sizeof(line), fp)) {
			ptr = strchr(line, '\n'); if (ptr) *ptr = 0;
			ptr = strchr(line, '\r'); if (ptr) *ptr = 0;
			ptr = strchr(line, delimiter);
			if (ptr) {
				*ptr = 0;
				ptr++;
				char *token = line;
				if (*token == '"') token++; // skip possible opening quote
				char *endquote = strchr(token, '"');
				if (endquote) *endquote = 0; // NULL-out end quote
				if (strcmp(entry, token) == 0) { // if this line's token identifier is the one we are looking for, then save the value
					if (*ptr == '"') ptr++; // skip possible opening quote
					endquote = strchr(ptr, '"');
					if (endquote) *endquote = 0;
					strncpy(buf, ptr, bufSize);
					buf[bufSize-1] = 0; // in case ptr string exceeded buf len
					ret = 0;
				}
			}
		}
		fclose(fp);
	}

	if (verbose > 0) {
		if (ret == 0)
			printf("Entry: %s Value: %s\n", entry, buf);
		else
			printf("Entry: %s not found?\n", entry);
	}

	return ret;
}

static void SendContactInfo(AeDRMDataItem *dataItem, const char* contactInfoFile)
{
	AeGetCurrentTime(&dataItem->value.timeStamp);

	// contactInfoFile format:
	// contactType contact info
	FILE *fp = fopen(contactInfoFile, "r");
	if (fp) {
		char line[1024];
		while (fgets(line, sizeof(line), fp)) {
			char *ptr;
			ptr = strchr(line, '\n'); if (ptr) *ptr = 0;
			ptr = strchr(line, '\r'); if (ptr) *ptr = 0;
			ptr = strchr(line, '\t');
			if (ptr) {
				*ptr = 0;
				char name[80];
				snprintf(name, sizeof(name), "TS.Contact.%s", line);
				name[sizeof(name) - 1] = 0;
				dataItem->pName = name;
				dataItem->value.data.pString = ++ptr;
				dataItem->value.iType = AeDRMDataString;
				dataItem->value.iQuality = AeDRMDataGood;
				AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
			}
		}
		fclose(fp);
	}

}

static void SendNetworkStatus(AeDRMDataItem *dataItem, const char* networkInfoFile)
{
	char info[4096];
	FILE *fp = fopen(networkInfoFile, "r");
	if (fp) {
		char line[1024];
		while (fgets(line, sizeof(line), fp)) {
			int lenInfo = strlen(info);
			strncat(info, line, sizeof(info) - lenInfo);
		}
		fclose(fp);
	}

	AeGetCurrentTime(&dataItem->value.timeStamp);
	dataItem->pName = "TS.Network";
	dataItem->value.data.pString = info;
	dataItem->value.iType = AeDRMDataString;
	dataItem->value.iQuality = AeDRMDataGood;
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
}

// experimentFileList is the name of a text file which contains
// the names of other files that contain experiment metrics
// to be sent to Axeda.  The file named by experimentFileList should have
// one experiment metrics file name per line.
static void SendExperimentMetrics(AeDRMDataItem *dataItem, const char* experimentFileList)
{
	FILE *explistfp = fopen(experimentFileList, "r");
	if (explistfp) {
		char expFileName[256];
		while (fgets(expFileName, sizeof(expFileName), explistfp)) {
			expFileName[strcspn(expFileName, "\n\r")] = '\0';
			FILE *expfp = fopen(expFileName, "r");
			if (!expfp)
				continue;

			char value[4096];
			if (fread(value, sizeof(char), sizeof(value), expfp))
			{
				AeGetCurrentTime(&dataItem->value.timeStamp);
				dataItem->pName = "TS.Experiment";
				dataItem->value.data.pString = value;
				dataItem->value.iType = AeDRMDataString;
				dataItem->value.iQuality = AeDRMDataGood;
				AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
			}

			fclose(expfp);

			char cmd[300];
			sprintf(cmd, "rm %s", expFileName);
			system(cmd);
		}

		fclose(explistfp);

		char cmd[300];
		sprintf(cmd, "rm %s", experimentFileList);
		system(cmd);
	}
}

void RSMInit()
{
	// make sure we have version info prior to init since we use the system-serial-number (dell service tag) as a hook
	GenerateVersionInfo();

	// grab our location once when we launch
	system("./location_helper.sh");

	// initialize the AgentInfo fields
	agentInfo.modelNumber = strdup("ION-TS1");

	char buf[256];
	int ret = GetConfigEntry(TS_VERSIONS, ':', "serialnumber", buf, sizeof(buf));
	if (ret != 0)
		ret = GetConfigEntry(ALT_SN, ':', "serialnumber", buf, sizeof(buf));
	if (ret != 0)
		ret = GetConfigEntry(TS_VERSIONS, ':', "host", buf, sizeof(buf));
	if (ret == 0) {
		if (strlen(buf) == 0)
			strcpy(buf, "unknown");
		agentInfo.serialNumber = strdup(buf);
	} else
		agentInfo.serialNumber = strdup("unknown");
	agentInfo.pingRate = pingRate;

	agentInfo.numPGMs = 0;
	agentInfo.pgmSerialNumberList = 0;

	time(&curTime); // gets time in seconds since 1970
}

void RSMClose()
{
	free(agentInfo.modelNumber);
	free(agentInfo.serialNumber);
	if (agentInfo.numPGMs > 0) {
		int i;
		for(i=0;i<agentInfo.numPGMs;i++)
			free(agentInfo.pgmSerialNumberList[i]);
		free(agentInfo.pgmSerialNumberList);
	}
}

void GetSoftwareVersion(char *softwareComponent, char *subcat, AeDRMDataItem *item, char *refFile)
{
	static char name[256];
	static char buf[256];
	buf[0] = 0;

	if (subcat)
		sprintf(name, "TS.%s.%s", subcat, softwareComponent);
	else
		sprintf(name, "TS.%s", softwareComponent);
	item->pName = name;
	item->value.data.pString = buf;
	item->value.iType = AeDRMDataString;
	item->value.iQuality = AeDRMDataGood;
	// some components are handled differently, most come from our versions file
	if (strcmp(softwareComponent, "TYPE") == 0) {
		strcpy(buf, "TS1");
	} else {
		int ret = GetConfigEntry(refFile, ':', softwareComponent, buf, sizeof(buf));
		if ( ret > 0)
			strcpy(buf, "unknown");
	}
}

void sigint_handler(int sig)
{
	printf("Got interrupt request, will process after timeout expires.\n");
	ok = 0;
}

int main(int argc, char *argv[])
{
	printf("RSM_TS Agent\n");
	printf("OPENSSL Version: %s\n", OPENSSL_VERSION_TEXT);

	char *site = 0;

	// process cmd-line args
	int argcc = 1;
	while (argcc < argc) {
		if (argv[argcc][0] == '-') {
			switch (argv[argcc][1]) {
				case 'v': // verbose bump
					verbose++;
				break;

				case 'c': // don't update config file
					wantConfigUpdate = 0;
				break;
			}
		} else {
			site = argv[argcc];
		}
		argcc++;
	}

	if (site == 0) {
		fprintf(stderr, "ERROR!  Usage: RSMAgent https://lifetechnologies-sandbox.axeda.com:443/eMessage\n");
		exit(0);
	}

	RSMInit();

	AeTimeValue pingRate, timeLimit;
	AeError rc;

	// initialize the Axeda embedded agent
	AeInitialize();

	// Axeda debug output on
	if (verbose > 1)
		AeDRMSetLogLevel(AeLogDebug);

	// set up a few options
	rc = AeWebSetSSL(AeWebCryptoMedium, AeFalse, NULL);
	if (rc != AeEOK) {
		fprintf(stderr, "Failed to set SSL parameters (%s)\n", AeGetErrorString(rc));
		return 1;
	}

	// configure master device
	rc = AeDRMAddDevice(AeDRMDeviceMaster, agentInfo.modelNumber, agentInfo.serialNumber, &iDeviceId);
	if (rc != AeEOK) {
		fprintf(stderr, "Failed to add device (%s)\n", AeGetErrorString(rc));
		return 1;
	}

	// configure primary DRM server
	pingRate.iSec = agentInfo.pingRate;
	pingRate.iMicroSec = 0;
	rc = AeDRMAddServer(AeDRMServerConfigPrimary, site, DEFAULT_OWNER, &pingRate, &iServerId);
	if (rc != AeEOK) {
		fprintf(stderr, "Failed to add server (%s)\n", AeGetErrorString(rc));
		return 1;
	}

	AeDRMSetOnFileUploadBegin(OnFileUploadBegin);
	AeDRMSetOnFileUploadData(OnFileUploadData);
	AeDRMSetOnFileUploadEnd(OnFileUploadEnd);

	printf("Initialized!\n");

	AeDRMDataItem dataItem;

	// install interrupt handler
	signal(SIGINT, sigint_handler);

	// main loop - sends heartbeat at ping rate, and notifies of any alarm conditions

	printf("Ready\n");

	ok = 1;
	while (ok) {
		time(&curTime);
		// in this loop, we check our list of updatable events and post any that trigger
		// we loop with a minimum granularity of one second, but most items update much less frequent
		unsigned int i;
		for(i=0;i<numUpdateItems;i++) {
			time_t deltaTime = curTime - updateItem[i].lastUpdateTime;
			if (deltaTime >= updateItem[i].updateRate) {
				updateItem[i].lastUpdateTime = curTime;
				if (UpdateDataItem(updateItem[i].type, &dataItem) > 0) {
					AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
					// here we can test on status items and see if they require an alarm be sent
/*
					AeDRMAlarm alarmData;
					if (UpdateAlarmItem((PGMStatusType)i, &alarmData, &dataItem) > 0) {
						AeDRMPostAlarm(iDeviceId, iServerId, AeDRMQueuePriorityUrgent, &alarmData);
printf ("ALARM!\n");
					}
*/
if (verbose > 0) {
	printf("Item: %s Value: ", dataItem.pName);
	if (dataItem.value.iType == AeDRMDataAnalog)
		printf("%.4lf\n", dataItem.value.data.dAnalog);
	else if (dataItem.value.iType == AeDRMDataString)
		printf("%s\n", dataItem.value.data.pString);
	else
		printf("?\n");
}
				}

			}
		}

		timeLimit.iSec = 1; // minimum poll rate
		timeLimit.iMicroSec = 0;
		AeDRMExecute(&timeLimit);
	}

	//  shutdown Axeda Agent Embedded 
	printf("Shutting down...\n");
	AeShutdown();
	RSMClose();

	if (fileSystemList != NULL)
		free(fileSystemList);

	printf("Done.\n");
	return 0;
}

void EventPostCB(void *user, char *name, int type, double val, char *info)
{
	AeDRMDataItem *dataItem = (AeDRMDataItem *)user;
	dataItem->pName = name;
	dataItem->value.iQuality = AeDRMDataGood;
	if (type == 0) {
		dataItem->value.iType = AeDRMDataAnalog;
		dataItem->value.data.dAnalog = val;
	} else {
		dataItem->value.iType = AeDRMDataString;
		dataItem->value.data.pString = info;
	}

	if (verbose > 0) {
		printf("Got log event: %s type: %d val: %.2lf info: %s\n", name, type, val, (info ? info : "none"));
	}

	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
}

int UpdateDataItem(StatusType status, AeDRMDataItem *dataItem)
{
	int ret = 0;

	AeGetCurrentTime(&dataItem->value.timeStamp);

	switch (status) {
		case STATUS_HD: {
			struct statfs buf;
			memset(&buf, 0, sizeof(buf));
			if (statfs("/results", &buf) == 0) {
				double newPercentFull = 1.0 - (double)buf.f_bavail / (double)buf.f_blocks;
				double delta = newPercentFull - percentFull;
				if (delta < 0) delta = -delta;
				if (delta > 0.01) {
					percentFull = newPercentFull;
					dataItem->pName = "TS.HW.HD.ResultsFull";
					dataItem->value.iType = AeDRMDataAnalog;
					dataItem->value.iQuality = AeDRMDataGood;
					dataItem->value.data.dAnalog = 100.0 * percentFull;
					ret = 1;
				}
			}
		} break;

		case STATUS_TASK:
			dataItem->pName = "TS.TASK";
			dataItem->value.iType = AeDRMDataString;
			dataItem->value.iQuality = AeDRMDataGood;
			dataItem->value.data.pString = "Idle"; // MGD placeholder
			ret = 1;
		break;

		case STATUS_EVENT:
			// here we check the event log for new entries, and post any we find via our callback
			ret = 0; // the caller will try and post if this is non-zero, so always return 0, we handle posts via the callback here
			// eventLogCheck(EventPostCB, dataItem);
		break;

		case STATUS_VERSIONS: {
			// check to see if the package database has been modified, if so we need to update our version information
			struct stat buf;
			memset(&buf, 0, sizeof(buf));
			stat("/var/lib/dpkg/status", &buf);
			if (buf.st_mtime != lastVersionModTime) {
				lastVersionModTime = buf.st_mtime;
				GenerateVersionInfo();
				SendVersionInfo(dataItem);
				// no need to set ret to 1, we have already posted the data
			}
		} break;

		case STATUS_CONTACTINFO: {
			// check to see if the contact info has been modified and if so update Axeda
			const char* contactsFile = "ContactInfo.txt";
			struct stat buf;
			memset(&buf, 0, sizeof(buf));
			int retVal = stat(contactsFile, &buf);
			if (retVal != 0)
			{
				// no contactinfo.txt so call updateContactInfo.py script here.
				// dbreports calls updateContactInfo.py if the contact info changes.
				// TODO: depends on getContactInfo.py putting contactinfo.txt where we can find it
				// which is currently in the same directory as the location of the script itself
				// which happens to be where the RSM agent is running from
				// TODO: contactsFile is not locked during write so script called from dbreports
				// could overwrite it while we are reading/writing it here
				retVal = system("python updateContactInfo.py");
				if (retVal == 0)
				{
					memset(&buf, 0, sizeof(buf));
					retVal = stat("ContactInfo.txt", &buf);
				}
			}
			if (retVal == 0 && buf.st_mtime != lastContactInfoTime) {
				lastContactInfoTime = buf.st_mtime;
				SendContactInfo(dataItem, contactsFile);
				// no need to set ret to 1, we have already posted the data
			}
		} break;

		case STATUS_SERVICES:
			system("python status.py > TSServers.txt");
			SendServersStatus(dataItem);
		break;

		case STATUS_FILESERVERS:
			system("python queryFileServers.py > TSFileServers.txt");
			SendFileServerStatus(dataItem);
		break;

		case STATUS_NETWORK:
			if (0 == system("TSquery > TSnetwork.txt"))
			{
				SendNetworkStatus(dataItem, "TSnetwork.txt");
			}
		break;

		case STATUS_EXPERIMENT:
		{
			// torrent server writes out the experiment metrics files at the end of analysis
			// (see /opt/ion/iondb/TLScript.py and RSM_TS/createExperimentMetrics.py)
			// if we see any experiment metrics files send them to DRM server
			// SendExperimentMetrics deletes the files when it is done with them.
			if (0 == system("ls -tr1 TSexperiment-*.txt > TSexperiments.txt"))
			{
				SendExperimentMetrics(dataItem, "TSexperiments.txt");
			}

		} break;

		case STATUS_PGMS:
			// get the list of attached PGM's
			// MGD - for now, we will just do this when we start up, but might want to do once an hour or something in the future
			// MGD - the python script stops after finding the first valid, could change in the future if we want the entire list
			//     - but the purpose initially was to allow Axeda to grab customer info from SAP using a serial number of a PGM

			// free old list
			if (agentInfo.numPGMs > 0) {
				int i;
				for(i=0;i<agentInfo.numPGMs;i++)
					free(agentInfo.pgmSerialNumberList[i]);
				free(agentInfo.pgmSerialNumberList);
				agentInfo.pgmSerialNumberList = NULL;
			}

			// build new list
			agentInfo.numPGMs = 0;
			agentInfo.pgmSerialNumberList = 0;
			system("python find_serial.py > PGM_list.txt");
			FILE *fp = fopen("PGM_list.txt", "rb");
			if (!fp)
				break;

			// count lines/PGMs
			agentInfo.numPGMs = 0;
			const int LF=10;
			int c;
			while ((c=fgetc(fp))!=EOF)
				agentInfo.numPGMs += (c==LF) ? 1 : 0; // one line per PGM serial number

			fseek(fp,0,SEEK_SET);

			agentInfo.pgmSerialNumberList = (char **)malloc(sizeof(char *)*agentInfo.numPGMs); // its a list of string pointers

			int i;
			for (i = 0; i < agentInfo.numPGMs; i++)
			{
				char serial[64], line[64];
				if (fgets(line, sizeof(line), fp) == NULL)
					break;

				sscanf(line, "%s", serial); // gets rid of LF
				if (serial[0] != 0) {
					char *ptr = serial;
					if (serial[0] == 's' && serial[1] == 'n')
						ptr = ptr + 2;
					agentInfo.pgmSerialNumberList[i] = strdup(ptr);
				}

				// send list
				if (i == 0) {
					dataItem->pName = "TS.PGM.Default"; // MGD - in the future, we can specify each PGM by name if we wanted to, but the 'Default' PGM can still be hooked into for SAP lookups
					dataItem->value.iQuality = AeDRMDataGood;
					dataItem->value.iType = AeDRMDataString;
					dataItem->value.data.pString = agentInfo.pgmSerialNumberList[i];
					AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
				}

				char itemName[64];
				sprintf(itemName, "TS.PGM.%d", i+1);
				dataItem->pName = itemName;
				dataItem->value.iQuality = AeDRMDataGood;
				dataItem->value.iType = AeDRMDataString;
				dataItem->value.data.pString = agentInfo.pgmSerialNumberList[i];
				AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
			}
			fclose(fp);

			// MGD - I'm leaving the ret code as 0 since I envision this code could send over multiple PGMs in the future, so I'm sending the one now
		break;
	}

	return ret;
}
 
void GenerateVersionInfo()
{
	if (wantConfigUpdate == 0)
		return;

	// first, generate our versions file:
	char cmd[1024];
	sprintf(cmd, "/opt/ion/iondb/bin/lversionChk.sh | sed {s/=/:/} > %s", TS_VERSIONS);
	system(cmd);
	sprintf(cmd, "cat /etc/torrentserver/tsconf.conf >> %s", TS_VERSIONS);
	system(cmd);
}

void SendServersStatus(AeDRMDataItem *dataItem)
{
	AeGetCurrentTime(&dataItem->value.timeStamp);
	GetSoftwareVersion("Crawler", "Server", dataItem, TS_SERVERS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("Archive", "Server", dataItem, TS_SERVERS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("Job", "Server", dataItem, TS_SERVERS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("Plugin", "Server", dataItem, TS_SERVERS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
}

void SendFileServerStatus(AeDRMDataItem *dataItem)
{
	AeGetCurrentTime(&dataItem->value.timeStamp);

	struct statfs buf;
	FILE *fp;
	fp = fopen("TSFileServers.txt", "r");
	if (fp) {
		char line[256];
		while (fgets(line, sizeof(line), fp)) {
			// remove trailing carriage returns, line feeds, slashes
			int len = strlen(line);
			int trim = 1;
			while (trim == 1 && len > 0) {
				trim = 0;
				if (line[len-1] == '\n') {
					line[len-1] = 0;
					len--;
					trim = 1;
				}
				if (line[len-1] == '\r') {
					line[len-1] = 0;
					len--;
					trim = 1;
				}
				if (line[len-1] == '/') {
					line[len-1] = 0;
					len--;
					trim = 1;
				}
			}
			memset(&buf, 0, sizeof(buf));
			if (statfs(line, &buf) == 0) {
				// construct the agent attribute name from the mounted file system name
				char *ptr = strrchr(line, '/'); // find the end of the name up to the last slash
				if (ptr == NULL)
					ptr = line; // if no slash, then just use the entire name from our input line
				// find file system in our list, add if not found
				int i;
				for(i=0;i<numFileSystems;i++) {
					if (strcmp(ptr, fileSystemList[i].agentAttributeName) == 0)
						break;
				}
				if (i == numFileSystems) { // not found, so add
					numFileSystems++;
					if (numFileSystems == 1)
						fileSystemList = (FileSystemList *)malloc(sizeof(FileSystemList));
					else
						fileSystemList = (FileSystemList *)realloc(fileSystemList, numFileSystems * sizeof(FileSystemList));
					strcpy(fileSystemList[i].agentAttributeName, ptr);
					strcpy(fileSystemList[i].mountedName, line);
					fileSystemList[i].percentFull = 0.0;
				}
				double newPercentFull = 1.0 - (double)buf.f_bavail / (double) buf.f_blocks;
				double delta = fabs(newPercentFull - fileSystemList[i].percentFull);
				if (delta > 0.01) {
					fileSystemList[i].percentFull = newPercentFull;
					char attribute[280];
					sprintf(attribute, "TS.HW.HD.%s", fileSystemList[i].agentAttributeName);
					dataItem->pName = attribute;
					dataItem->value.iType = AeDRMDataAnalog;
					dataItem->value.iQuality = AeDRMDataGood;
					dataItem->value.data.dAnalog = 100.0 * fileSystemList[i].percentFull;
					AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
				}
			}
		}
		fclose(fp);
	}
}

void SendVersionInfo(AeDRMDataItem *dataItem)
{
	// now look through that file for various informational items
	AeGetCurrentTime(&dataItem->value.timeStamp);
	GetSoftwareVersion("TYPE", NULL, dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("analysis", "Version", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("alignment", "Version", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("dbreports", "Version", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("tmap", "Version", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("docs", "Version", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("tsconfig", "Version", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("referencelibrary", "Version", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("host", NULL, dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("hostname", "Config", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("ipaddress", "Config", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("mode", "Config", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("configuration", "Config", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("serialnumber", "Config", dataItem, TS_VERSIONS);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);

/*
	GetSoftwareVersion("country", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("region", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("code", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("city", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("ipaddress", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
*/

	GetSoftwareVersion("State", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("City", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("Street-Address", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("Org-Name", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	GetSoftwareVersion("Postal-Code", "Location", dataItem, LOC_FILE);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
}
 
/******************************************************************************
 * Callbacks
 ******************************************************************************/

/******************************************************************************/
static AeBool OnFileUploadBegin(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AePointer *ppUserData)
{
    AeDemoUpload *pUpload;

    /* allocate upload descriptor */
    pUpload = (AeDemoUpload *) malloc(sizeof(AeDemoUpload));
    if (!pUpload)
        return AeFalse;

    /* initialize */
    memset(pUpload, 0, sizeof(AeDemoUpload));
    pUpload->ppUploads = ppUploads;
    pUpload->iUploadIdx = 0;
    pUpload->iCurFileHandle = AeFileInvalidHandle;

    *ppUserData = pUpload;

    return AeTrue;
}

/******************************************************************************/
static AeBool OnFileUploadData(AeInt32 iDeviceId, AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData)
{
    AeDemoUpload *pUpload;

    *ppFile = NULL;
    *ppData = NULL;
    *piSize = 0;

    pUpload = (AeDemoUpload *) pUserData;
    if (!pUpload)
        return AeFalse;

    /* no more files to upload: indicate that */
    if (!pUpload->ppUploads[pUpload->iUploadIdx])
        return AeTrue;

    /* initialize next file */
    if (pUpload->iCurFileHandle == AeFileInvalidHandle)
    {
	/* inspect file name and perform special actions prior to file delivery */
	int rsshfile = 0;
	if (strstr(pUpload->ppUploads[pUpload->iUploadIdx]->pName, "rssh") != 0) {
		// parse out the info from the 'command' file name
		// expected file names are:
		//    rssh-start-rsshUser-i0nrssh
		//    rssh-stop-dontcare-dontcare
		char *rsshCmd;
		char *user;
		char *pass;
		char tokens[1024];
		char cmd[1024];
		strncpy(tokens, pUpload->ppUploads[pUpload->iUploadIdx]->pName, sizeof(cmd));
		strtok(tokens, "-"); /* we don't care about the /rssh- part */
		rsshCmd = strtok(NULL, "-");
		user = strtok(NULL, "-");
		pass = strtok(NULL, "-");

		// execute the rssh command
		if (rsshCmd && pass && user) {
			srand(time(NULL));
			nextPort = 15000 + (rand() % 1024);

			// create a file containing the port that will be uploaded.  The caller will need this port information to connect!
			FILE *fp = fopen("/tmp/rsshcmd", "w");
			// fprintf(fp, "RSSH command: %s to rssh.iontorrent.net on port %d\n", rsshCmd, nextPort);
			fprintf(fp, "ssh rssh.iontorrent.net as rsshUser, then run:\nssh -l ionadmin -p %d -o NoHostAuthenticationForLocalhost=yes -o StrictHostKeyChecking=no localhost", nextPort);
			fclose(fp);

			sprintf(cmd, "script -c \"./reverse_ssh.sh %s 22 22 %d rssh.iontorrent.net %s %s\" /dev/null &", rsshCmd, nextPort, user, pass);
			if (verbose > 0)
				printf("System cmd executing: %s\n", cmd);
			system(cmd);

			rsshfile = 1;
		}
	}

        /* open file */
	if (rsshfile == 1)
        	pUpload->iCurFileHandle = AeFileOpen("/tmp/rsshcmd", AE_OPEN_READ_ONLY);
	else
        	pUpload->iCurFileHandle = AeFileOpen(pUpload->ppUploads[pUpload->iUploadIdx]->pName, AE_OPEN_READ_ONLY);
        if (pUpload->iCurFileHandle == AeFileInvalidHandle)
            return AeFalse;

        pUpload->curFileStat.pName = pUpload->ppUploads[pUpload->iUploadIdx]->pName;
        pUpload->curFileStat.iType = AeFileTypeRegular;
	if (rsshfile == 1) {
        	pUpload->curFileStat.iSize =
#ifndef ENABLE_LARGEFILE64
            	AeFileGetSize
#else
            	AeFileGetSize64
#endif
            	("/tmp/rsshcmd");
	} else {
        	pUpload->curFileStat.iSize =
#ifndef ENABLE_LARGEFILE64
            	AeFileGetSize
#else
            	AeFileGetSize64
#endif
            	(pUpload->ppUploads[pUpload->iUploadIdx]->pName);
	}
        pUpload->curFileStat.iMTime = 0;
    }

    *ppFile = &pUpload->curFileStat;

    /* try to read another portion of the file */
    *piSize = AeFileRead(pUpload->iCurFileHandle, pUpload->pBuffer, sizeof(pUpload->pBuffer));
    if (*piSize < 0)
        return AeFalse;
    else if (*piSize == 0)
    {
        AeFileClose(pUpload->iCurFileHandle);
        pUpload->iCurFileHandle = AeFileInvalidHandle;

        if (pUpload->ppUploads[pUpload->iUploadIdx]->bDelete)
            AeFileDelete(pUpload->ppUploads[pUpload->iUploadIdx]->pName);

        pUpload->iUploadIdx += 1;
    }
    else if (*piSize > 0)
        *ppData = pUpload->pBuffer;

    return AeTrue;
}

/******************************************************************************/
static void OnFileUploadEnd(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData)
{
    AeDemoUpload *pUpload;

    pUpload = (AeDemoUpload *) pUserData;
    if (!pUpload)
        return;

    if (pUpload->iCurFileHandle != AeFileInvalidHandle)
        AeFileClose(pUpload->iCurFileHandle);

    AeFree(pUpload);
}
