//
// Remote Support and Monitor Agent - Torrent Server
// (c) 2011 Life Technologies, Ion Torrent
//

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/vfs.h>
#include <stdlib.h>
#include <openssl/opensslv.h>

#include "AeOSLocal.h"
#include "AeTypes.h"
#include "AeError.h"
#include "AeOS.h"
#include "AeInterface.h"

#include "parson.h"

#include "AlarmMgr.h"

#define DEFAULT_OWNER "drm-data_source"

#define BIN_DIR "/opt/ion/RSM"
#define SPOOL_DIR "/var/spool/ion"
#define TS_VERSIONS		SPOOL_DIR "/TSConfig.txt"
#define TS_SERVERS		SPOOL_DIR "/TSServers.txt"
#define TS_NETWORK		SPOOL_DIR "/TSnetwork.txt"
#define TS_FILESERVERS	SPOOL_DIR "/TSFileServers.txt"
#define TS_EXPERIMENTS	SPOOL_DIR "/TSexperiments.txt"
#define TS_CONTACTINFO	SPOOL_DIR "/ContactInfo.txt"
#define INSTR_LIST		SPOOL_DIR "/InstrumentList.txt"
#define ALT_SN			SPOOL_DIR "/serial_number.alt"
#define LOC_FILE		SPOOL_DIR "/loc.txt"
#define RAIDINFO_FILE	SPOOL_DIR "/raidstatus.json"

typedef enum {
	STATUS_HD = 0,
	STATUS_EVENT,
	STATUS_VERSIONS,
	STATUS_SERVICES,
	STATUS_FILESERVERS,
	STATUS_INSTRS,
	STATUS_CONTACTINFO,
	STATUS_NETWORK,
	STATUS_EXPERIMENT,
	STATUS_HARDWARE,
	STATUS_RAIDINFO
} StatusType;

typedef struct {
	StatusType	type;
	time_t		updateRate;
	time_t		lastUpdateTime;
} UpdateItem;

typedef struct {
	char	*modelNumber;
	char	*serialNumber;
	int		pingRate;
	int		numInsts;
	char	**serialNumberList;
} AgentInfo;

typedef struct _AeDemoUpload AeDemoUpload;

struct _AeDemoUpload {
	AeFileUploadSpec **ppUploads;
	AeInt32          iUploadIdx;
	AeFileStat       curFileStat;
	AeFileHandle     iCurFileHandle;
	AeChar           pBuffer[BUFSIZ];
};

#define LINE_LENGTH 64
#define CMD_LENGTH 256
typedef struct webProxy_s {
	int useProxy;
	AeChar pHost[LINE_LENGTH];
	AeUInt16 iPort;
	AeChar pUser[LINE_LENGTH];
	AeChar pPass[LINE_LENGTH];
} webProxy_t;

// -- File system list management --
typedef struct {
	char	mountedName[256];
	char	agentAttributeName[256]; // last portion of mounted name without trailing slash
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
		{STATUS_INSTRS, 3600, 0},
		//{STATUS_NETWORK, 3600, 0},
		{STATUS_HARDWARE, 3600, 0},
		{STATUS_RAIDINFO,  600, 0},
};
unsigned int numUpdateItems = sizeof(updateItem) / sizeof(UpdateItem);
webProxy_t proxyInfo;

int UpdateDataItem(StatusType status, AeDRMDataItem *dataItem);
void SendVersionInfo(AeDRMDataItem *dataItem);
void SendServersStatus(AeDRMDataItem *dataItem);
void GenerateVersionInfo();
void SendFileServerStatus(AeDRMDataItem *dataItem);
void checkEnvironmentForWebProxy (webProxy_t *proxyInfo);
char *getTextUpToDelim(char *start, char delim, char *output, int outputSize);
void readAlarmsFromJson(char const * const filename);
void buildRaidAlarmName(const int eNum, const int sNum, char const * const iName,
		char * aName, size_t aNameSize);
int  numericValue(char const * const value, double *number);
void WriteAeStringDataItem(char const * const subcat, char const * const key,
		char const * const value, AeDRMDataItem *item);
void WriteAeAnalogDataItem(char const * const subcat, char const * const key,
		double value, AeDRMDataItem *item);

// file upload callbacks
static AeBool OnFileUploadBegin(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AePointer *ppUserData);
static AeBool OnFileUploadData(AeInt32 iDeviceId, AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData);
static void OnFileUploadEnd(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData);

void trimTrailingWhitespace(char *inputBuf)
{
	const char space = 32;
	unsigned int length;
	if (!inputBuf)
		return;

	length = strlen(inputBuf);
	while (length && inputBuf[length-1] <= space)
		inputBuf[--length] = '\0';
}       // end trimTrailingWhitespace

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
	int rc;
	FILE *explistfp = fopen(experimentFileList, "r");
	if (explistfp) {
		char expFileName[256];
		while (fgets(expFileName, sizeof(expFileName), explistfp)) {
			expFileName[strcspn(expFileName, "\n\r")] = '\0';
			FILE *expfp = fopen(expFileName, "r");
			if (!expfp)
				continue;

			char value[4096] = {0};
			if (fread(value, sizeof(char), sizeof(value), expfp)) {
				AeGetCurrentTime(&dataItem->value.timeStamp);
				dataItem->pName = "TS.Experiment";
				dataItem->value.data.pString = value;
				dataItem->value.iType = AeDRMDataString;
				dataItem->value.iQuality = AeDRMDataGood;
				AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
			}

			fclose(expfp);

			char cmd[300];
			snprintf(cmd, 300, "rm %s", expFileName);
			rc = system(cmd);
			if (rc == -1)
				printf("%s: system: %s\n", __FUNCTION__, strerror(errno));
		}

		fclose(explistfp);

		char cmd[300];
		snprintf(cmd, 300, "rm %s", experimentFileList);
		rc = system(cmd);
		if (rc == -1)
			printf("%s: system: %s\n", __FUNCTION__, strerror(errno));
	}
}

static void SendHardwareName(AeDRMDataItem *dataItem, const char* nameFile)
{
	char buf[256];
	int ret;
	ret = GetConfigEntry((char*) nameFile, ':', "hardwarename", buf, sizeof(buf));
	if (ret == 0)
	{
		AeGetCurrentTime(&dataItem->value.timeStamp);
		dataItem->pName = "TS.Config.hwname";
		dataItem->value.data.pString = buf;
		dataItem->value.iType = AeDRMDataString;
		dataItem->value.iQuality = AeDRMDataGood;
		AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	}
}

static void SendBiosVersion(AeDRMDataItem *dataItem, const char* nameFile)
{
	char buf[256];
	int ret;
	ret = GetConfigEntry((char*) nameFile, ':', "biosversion", buf, sizeof(buf));
	if (ret == 0)
	{
		AeGetCurrentTime(&dataItem->value.timeStamp);
		dataItem->pName = "TS.Config.biosversion";
		dataItem->value.data.pString = buf;
		dataItem->value.iType = AeDRMDataString;
		dataItem->value.iQuality = AeDRMDataGood;
		AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
	}
}

void RSMInit()
{
	// make sure we have version info prior to init since we use the system-serial-number (dell service tag) as a hook
	GenerateVersionInfo();

	// grab our location once when we launch
	int rc = system(BIN_DIR "/location_helper.sh");
	if (rc == -1)
		printf("%s: system: %s\n", __FUNCTION__, strerror(errno));

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

	agentInfo.numInsts = 0;
	agentInfo.serialNumberList = 0;

	time(&curTime); // gets time in seconds since 1970

	checkEnvironmentForWebProxy(&proxyInfo);
}

void RSMClose()
{
	free(agentInfo.modelNumber);
	free(agentInfo.serialNumber);
	if (agentInfo.numInsts > 0) {
		int i;
		for(i=0;i<agentInfo.numInsts;i++)
			free(agentInfo.serialNumberList[i]);
		free(agentInfo.serialNumberList);
	}
}

void GetSoftwareVersion(char *softwareComponent, char *subcat, AeDRMDataItem *item, char *refFile)
{
	static char name[256];
	static char buf[256];
	buf[0] = 0;

	if (subcat)
		snprintf(name, 256, "TS.%s.%s", subcat, softwareComponent);
	else
		snprintf(name, 256, "TS.%s", softwareComponent);
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

void sigint_handler(int sig __attribute__((unused)))
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

	if (proxyInfo.useProxy) {
		printf("Using web proxy: host:%s port:%d\n", proxyInfo.pHost, proxyInfo.iPort);
		rc = AeWebSetProxy(AeWebProxyProtoHTTP, proxyInfo.pHost, proxyInfo.iPort, proxyInfo.pUser, proxyInfo.pPass);
		if (rc == AeEOK)
			printf("Web proxy was set successfully.\n");
		else
			fprintf(stderr, "Failed to set proxy, connecting directly: %s\n", AeGetErrorString(rc));
	}

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

	AlarmMgrInit(iDeviceId, iServerId);

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
	AlarmMgrSave();
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
	char cmd[CMD_LENGTH];
	int ret = 0;
	int rc;

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
		struct stat buf;
		memset(&buf, 0, sizeof(buf));
		int retVal = stat(TS_CONTACTINFO, &buf);
		if (retVal != 0) {
			// no TS_CONTACTINFO file so call updateContactInfo.py script here.
			// dbreports calls updateContactInfo.py if the contact info changes.
			// TODO: contactsFile is not locked during write so script called from dbreports
			// could overwrite it while we are reading/writing it here
			snprintf(cmd, CMD_LENGTH, "python %s/updateContactInfo.py", BIN_DIR);
			retVal = system(cmd);
			if (retVal == 0) {
				memset(&buf, 0, sizeof(buf));
				retVal = stat(TS_CONTACTINFO, &buf);
			}
		}
		if (retVal == 0 && buf.st_mtime != lastContactInfoTime) {
			lastContactInfoTime = buf.st_mtime;
			SendContactInfo(dataItem, TS_CONTACTINFO);
			// no need to set ret to 1, we have already posted the data
		}
	} break;

	case STATUS_SERVICES:
		snprintf(cmd, CMD_LENGTH, "python %s/status.py > %s", BIN_DIR, TS_SERVERS);
		rc = system(cmd);
		if (rc == -1)
			printf("%s: system: %s\n", __FUNCTION__, strerror(errno));
		SendServersStatus(dataItem);
		break;

	case STATUS_FILESERVERS:
		snprintf(cmd, CMD_LENGTH, "python %s/queryFileServers.py > %s", BIN_DIR, TS_FILESERVERS);
		rc = system(cmd);
		if (rc == -1)
			printf("%s: system: %s\n", __FUNCTION__, strerror(errno));
		SendFileServerStatus(dataItem);
		break;

	case STATUS_NETWORK:
		snprintf(cmd, CMD_LENGTH, "TSquery > %s", TS_NETWORK);
		if (0 == system(cmd)) {
			SendNetworkStatus(dataItem, TS_NETWORK);
		}
		break;

	case STATUS_EXPERIMENT:
		// torrent server writes out the experiment metrics files at the end of analysis
		// (see dbReports/iondb/anaserve/serve.py)
		// if we see any experiment metrics files send them to DRM server
		// SendExperimentMetrics deletes the files when it is done with them.
		snprintf(cmd, CMD_LENGTH, "ls -tr1 %s/TSexperiment-*.txt > %s", SPOOL_DIR, TS_EXPERIMENTS);
		if (0 == system(cmd)) {
			SendExperimentMetrics(dataItem, TS_EXPERIMENTS);
		}
	break;

	case STATUS_HARDWARE:
		// Identifies the type of computer hardware the server is running on.
		// Dell T7500 or Dell T620 for example.
		// product_info.alt file is written during RSM_launch startup script execution
		SendHardwareName(dataItem, "product_info.alt");
		// Records version of BIOS installed on server
		SendBiosVersion(dataItem, "product_info.alt");
	break;

	case STATUS_INSTRS:
		// get the list of attached instruments

		// free old list
		if (agentInfo.numInsts > 0) {
			int i;
			for(i=0;i<agentInfo.numInsts;i++)
				free(agentInfo.serialNumberList[i]);
			free(agentInfo.serialNumberList);
			agentInfo.serialNumberList = NULL;
		}

		// build new list
		agentInfo.numInsts = 0;
		agentInfo.serialNumberList = 0;
		snprintf(cmd, CMD_LENGTH, "python %s/find_serial.py > %s", BIN_DIR, INSTR_LIST);
		rc = system(cmd);
		if (rc == -1)
			printf("%s: system: %s\n", __FUNCTION__, strerror(errno));
		FILE *fp = fopen(INSTR_LIST, "rb");
		if (!fp)
			break;

		// count lines/instruments
		agentInfo.numInsts = 0;
		const int LF=10;
		int c;
		while ((c=fgetc(fp))!=EOF)
			agentInfo.numInsts += (c==LF) ? 1 : 0; // one line per instrument serial number

		fseek(fp,0,SEEK_SET);

		agentInfo.serialNumberList = (char **)malloc(sizeof(char *)*agentInfo.numInsts); // its a list of string pointers

		int i;
		for (i = 0; i < agentInfo.numInsts; i++) {
			char serial[64], line[64];
			if (fgets(line, sizeof(line), fp) == NULL)
				break;

			sscanf(line, "%s", serial); // gets rid of LF
			if (serial[0] != 0) {
				char *ptr = serial;
				if (serial[0] == 's' && serial[1] == 'n')
					ptr = ptr + 2;
				agentInfo.serialNumberList[i] = strdup(ptr);
			}

			// send list
			if (i == 0) {
				dataItem->pName = "TS.PGM.Default"; // can specify each instr. by name, but 'Default' instr. can still be hooked into for SAP lookups
				dataItem->value.iQuality = AeDRMDataGood;
				dataItem->value.iType = AeDRMDataString;
				dataItem->value.data.pString = agentInfo.serialNumberList[i];
				AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
			}

			char itemName[64];
			snprintf(itemName, 64, "TS.PGM.%d", i+1);
			dataItem->pName = itemName;
			dataItem->value.iQuality = AeDRMDataGood;
			dataItem->value.iType = AeDRMDataString;
			dataItem->value.data.pString = agentInfo.serialNumberList[i];
			AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
		}
		fclose(fp);

	case STATUS_RAIDINFO:
		readAlarmsFromJson(RAIDINFO_FILE);
		break;
	}

	return ret;
}

void GenerateVersionInfo()
{
	int rc;
	if (wantConfigUpdate == 0)
		return;

	// first, generate our versions file:
	char cmd[1024];

	snprintf(cmd, 1024, "ion_versionCheck.py | sed {s/=/:/} > %s", TS_VERSIONS);
	rc = system(cmd);
	if (rc == -1)
		printf("%s: system: %s\n", __FUNCTION__, strerror(errno));

	snprintf(cmd, 1024, "cat /etc/torrentserver/tsconf.conf >> %s", TS_VERSIONS);
	rc = system(cmd);
	if (rc == -1)
		printf("%s: system: %s\n", __FUNCTION__, strerror(errno));
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
	fp = fopen(TS_FILESERVERS, "r");
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
					snprintf(attribute, 280, "TS.HW.HD.%s", fileSystemList[i].agentAttributeName);
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

void checkEnvironmentForWebProxy (webProxy_t *proxyInfo)
{
	FILE *fp;
	char line[LINE_LENGTH];

	// initialize web proxy struct
	proxyInfo->useProxy =   0;
	proxyInfo->pHost[0] = '\0';
	proxyInfo->iPort    =   0;
	proxyInfo->pUser[0] = '\0';
	proxyInfo->pPass[0] = '\0';

	fp = fopen("/etc/environment", "r");
	if (!fp)
		return;

	while (fgets(line, LINE_LENGTH, fp)) {
		// sample http_proxy line:
		// http_proxy=http://user:pass@1.2.3.4:5
		const char *token = "http_proxy=http://";
		const int tlen = strlen(token);
		if (strstr(line, token)) {
			char buf[LINE_LENGTH];
			char *pos = line + tlen;

			pos = getTextUpToDelim(pos, ':', proxyInfo->pUser, LINE_LENGTH);
			pos = getTextUpToDelim(pos, '@', proxyInfo->pPass, LINE_LENGTH);
			pos = getTextUpToDelim(pos, ':', proxyInfo->pHost, LINE_LENGTH);
			pos = getTextUpToDelim(pos, '\0', buf, LINE_LENGTH);

			proxyInfo->iPort = atoi(buf);
			proxyInfo->useProxy = 1;
			break;
		}
	}
	fclose(fp);
}

char *getTextUpToDelim(char *start, char delim, char *output, int outputSize)
{
	char *pos;
	int   ii = 0;
	char *end = strchr(start, delim);
	if (!end)
		end = output + strlen(output);

	for (pos = start; pos < end && *pos != '\n' && ii < outputSize; ++pos)
		output[ii++] = *pos;
	output[ii] = '\0';

	return end + 1; // so next search can pick up where this one left off
}

// Read raidstatus.json and identify the items with non-blank, non-good status
void readAlarmsFromJson(char const * const filename)
{
	JSON_Value *root_value;
	JSON_Array *raid_status;
	JSON_Array *drives;
	JSON_Object *obj;
	size_t ii, jj;
	int encNum = -1;
	char alarmName[LINE_LENGTH];

	// Read the raidstatus.json file into memory.
	root_value = json_parse_file(filename);
	if (!root_value)
		return;

	obj = json_value_get_object(root_value);
	if (!obj) {
		json_value_free(root_value);
		return;
	}

	//const char *date = json_object_get_string(obj, "date");
	raid_status = json_object_get_array(obj, "raid_status");
	if (!obj) {
		json_value_free(root_value);
		return;
	}

	obj = json_array_get_object(raid_status, 0);
	if (!obj) {
		json_value_free(root_value);
		return;
	}

	const char *encId = json_object_get_string(obj, "enclosure_id");
	encNum = atoi(encId);

	drives = json_object_get_array(obj, "drives");
	if (!drives) {
		json_value_free(root_value);
		return;
	}

	// For each drive in raidstatus.json:
	for (ii = 0; ii < json_array_get_count(drives); ii++) {
		int slotNum = -1;

		obj = json_array_get_object(drives, ii);
		if (!obj)
			continue;

		JSON_Array *info = json_object_get_array(obj, "info");
		if (!info)
			continue;

		// For each drive parameter:
		for (jj = 0; jj < json_array_get_count(info); jj++) {

			JSON_Array *drvinf = json_array_get_array(info, jj);
			if (!drvinf)
				continue;

			const char *itemName = json_array_get_string(drvinf, 0);
			const char *itemDesc = json_array_get_string(drvinf, 1);
			const char *passFail = json_array_get_string(drvinf, 2);

			if (0 == strcmp(itemName, "Slot"))
				slotNum = atoi(itemDesc);

			// Items like "Slot" and "Inquiry Data", for which no alarm conditions exist,
			// have a passFail field with length 0.
			if ((!passFail) ||					// error
					(0 == strlen(passFail)))	// items for which no alarms conditions exist
				continue;

			buildRaidAlarmName(encNum, slotNum, itemName, alarmName, LINE_LENGTH);

			if (0 != strcmp("good", passFail)) {
				AlarmMgr_AddAlarmByName(alarmName, itemDesc);
			}
			else {
				AlarmMgr_DelAlarmByName(alarmName);
			}
		}

	}

	json_value_free(root_value);
}	// end readAlarmsFromJson

void buildRaidAlarmName(const int encNum, const int slotNum,
		char const * const itemName, char * alarmName, size_t alarmNameSize)
{
	size_t itemNameLen;
	size_t oi ;
	size_t ii; // input index

	if (!itemName || !alarmName || !alarmNameSize)
		return;

	itemNameLen = strlen(itemName);
	memset(alarmName, 0, alarmNameSize);
	snprintf(alarmName, alarmNameSize, "Alarm.RaidEnc%d.Slot%d.", encNum, slotNum);
	oi = strlen(alarmName); // output index

	for (ii = 0; ii < itemNameLen && oi < alarmNameSize - 1; ++ii) {
		if (itemName[ii] != ' '  && itemName[ii] != '.') {
			alarmName[oi++] = itemName[ii];
		}
	}
}	// end buildRaidAlarmName

int numericValue(char const * const value, double *number)
{
	int success = 0;
	int usedEntireString = 0;
	if ((!value) || !number)
		return 0; // fail

	errno = 0;
	char *ptrToLastCharUsed = NULL;
	*number = strtod(value, &ptrToLastCharUsed);

	char const * const ptrToEndOfValue = value + (strlen(value));

	usedEntireString = ptrToEndOfValue <= ptrToLastCharUsed;

	if (errno == 0 && usedEntireString)
		success = 1;

	return success;
}	// end numericValue

void WriteAeStringDataItem(char const * const subcat, char const * const key,
		char const * const value, AeDRMDataItem *item)
{
	static char name[256] = {0};
	if (subcat)
		snprintf(name, 256, "%s.%s", subcat, key);
	else
		snprintf(name, 256, "%s", key);
	item->pName = name;
	item->value.data.pString = (char *)value;
	item->value.iType = AeDRMDataString;
	item->value.iQuality = AeDRMDataGood;
}	// end WriteAeStringDataItem

void WriteAeAnalogDataItem(char const * const subcat, char const * const key,
		double value, AeDRMDataItem *item)
{
	static char name[2048] = {0};
	if (subcat)
		snprintf(name, 2048, "%s.%s", subcat, key);
	else
		snprintf(name, 2048, "%s", key);
	item->pName = name;
	item->value.data.dAnalog = value;
	item->value.iType = AeDRMDataAnalog;
	item->value.iQuality = AeDRMDataGood;
}	// end WriteAeAnalogDataItem


/******************************************************************************
 * Callbacks
 ******************************************************************************/

/******************************************************************************/
static AeBool OnFileUploadBegin(AeInt32 iDeviceId __attribute__((unused)), 
		AeFileUploadSpec **ppUploads, AePointer *ppUserData)
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
static AeBool OnFileUploadData(AeInt32 iDeviceId __attribute__((unused)),
		AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData)
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
	if (pUpload->iCurFileHandle == AeFileInvalidHandle)	{
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
			int rc;
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

				snprintf(cmd, 1024, "script -c \"./reverse_ssh.sh %s 22 22 %d rssh.iontorrent.net %s %s\" /dev/null &", rsshCmd, nextPort, user, pass);
				if (verbose > 0)
					printf("System cmd executing: %s\n", cmd);
				rc = system(cmd);
				if (rc == -1)
					printf("%s: system: %s\n", __FUNCTION__, strerror(errno));

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
static void OnFileUploadEnd(AeInt32 iDeviceId __attribute__((unused)),
		AeBool bOK __attribute__((unused)), AePointer pUserData)
{
	AeDemoUpload *pUpload;

	pUpload = (AeDemoUpload *) pUserData;
	if (!pUpload)
		return;

	if (pUpload->iCurFileHandle != AeFileInvalidHandle)
		AeFileClose(pUpload->iCurFileHandle);

	AeFree(pUpload);
}
