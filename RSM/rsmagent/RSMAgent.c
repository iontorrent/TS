//
// Remote Support and Monitor Agent - Torrent Server
// (c) 2011 Life Technologies, Ion Torrent
//
#define _BSD_SOURCE
// for sigprocmask
#define _POSIX_SOURCE

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
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
#include "sshTunnelMgmt.h"

#define DEFAULT_OWNER "drm-data_source"

#define BIN_DIR "/opt/ion/RSM"
#define SPOOL_DIR "/var/spool/ion"
#define TS_VERSIONS		SPOOL_DIR "/TSConfig.txt"
#define TS_SERVERS		SPOOL_DIR "/TSServers.txt"
#define TS_SERVERS_OLD	SPOOL_DIR "/TSServers.old"
#define TS_NETWORK		SPOOL_DIR "/TSnetwork.txt"
#define TS_FILESERVERS	SPOOL_DIR "/TSFileServers.txt"
#define TS_EXPERIMENTS	SPOOL_DIR "/TSexperiments.txt"
#define TS_CONTACTINFO	SPOOL_DIR "/ContactInfo.txt"
#define INSTR_LIST		SPOOL_DIR "/InstrumentList.txt"
#define ALT_SN			SPOOL_DIR "/serial_number.alt"
#define LOC_FILE		SPOOL_DIR "/loc.txt"
#define RAIDINFO_FILE	SPOOL_DIR "/raidstatus.json"
#define LINELEN 128
#define RSSH_RESULTS_FILE "/tmp/rsshResults"
#define RSSH_CMD_FILE "/tmp/rsshcmd"
#define OS_VERSION_SIZE 32

typedef enum {
	STATUS_HD = 0,
	STATUS_EVENT,
	STATUS_VERSIONS,
	STATUS_OS_VERSION,
	STATUS_SERVICES,
	STATUS_FILESERVERS,
	STATUS_INSTRS,
	STATUS_CONTACTINFO,
	STATUS_LOCATIONINFO,
	STATUS_NETWORK,
	STATUS_EXPERIMENT,
	STATUS_RAIDINFO,
	STATUS_GPU
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

// Server status queries
typedef struct serverStatusItem_s {
	char *name;
	char *status;
} serverStatusItem_t;

typedef struct serverStatus_s {
	int count;
	serverStatusItem_t *server;
	size_t size;
} serverStatus_t;

// globals
AgentInfo agentInfo;
int running = 1;
int verbose = 0;
unsigned long pingRate = 30;
time_t curTime, timeNow;
AeInt32 iDeviceId, iServerId;
static double percentFull = -1.0;
static time_t lastVersionModTime = 0;
static time_t lastContactInfoTime = 0;
static time_t lastLocationInfoTime = 0;
static int wantConfigUpdate = 1;
static char const * const gpuErrorFile = "/var/spool/ion/gpuErrors";
static char savedOsVersion[LINELEN] = {0};

UpdateItem updateItem[] = {
		{STATUS_VERSIONS, 120, 0},		// check every 2 min but only send updates if versions have changed
		{STATUS_CONTACTINFO, 120, 0},	// check every 2 min but only send updates if contact info has changed
		{STATUS_LOCATIONINFO,120, 0},	// check every 2 min but only send updates if location info has changed
		{STATUS_EXPERIMENT, 120, 0},
		{STATUS_SERVICES, 360, 0},		// send status of all servers every 6 minutes, even if no change
		{STATUS_FILESERVERS, 600, 0},
		{STATUS_INSTRS, 3600, 0},
		{STATUS_OS_VERSION, 3600, 0},
		//{STATUS_NETWORK, 3600, 0},
		{STATUS_RAIDINFO,  600, 0},
		{STATUS_GPU, 120, 0},
};
unsigned int numUpdateItems = sizeof(updateItem) / sizeof(UpdateItem);
webProxy_t proxyInfo;
static pthread_t locationThreadId = 0;
static int locationThreadExited = 0;

int UpdateDataItem(StatusType status, AeDRMDataItem *dataItem);
void SendVersionInfo(AeDRMDataItem *dataItem);
void SendServersStatus(AeDRMDataItem *dataItem);
void GenerateVersionInfo();
void SendFileServerStatus(AeDRMDataItem *dataItem);
void checkEnvironmentForWebProxy();
void readAlarmsFromJson(char const * const filename);
void buildRaidAlarmName(const int eNum, const int sNum, char const * const iName,
		char * aName, size_t aNameSize);
void WriteAeStringDataItem(char const * const subcat, char const * const key,
		char const * const value, AeDRMDataItem *item);
void readGpuErrorFile(char const * const filename, int *gpuFound, int *allRevsValid);

// file upload callbacks
static AeBool OnFileUploadBegin(AeInt32 iDeviceId, AeFileUploadSpec **ppUploads, AePointer *ppUserData);
static AeBool OnFileUploadData(AeInt32 iDeviceId, AeFileStat **ppFile, AeChar **ppData, AeInt32 *piSize, AePointer pUserData);
static void OnFileUploadEnd(AeInt32 iDeviceId, AeBool bOK, AePointer pUserData);
static void SendLocationInfo(AeDRMDataItem *dataItem);
void *locationThreadTask(void *args __attribute__((unused)));
int ReadOsVersion(char *OsVersion, const size_t OsVersionSize);

void trimTrailingWhitespace(char *inputBuf)
{
	const char space = 32;
	unsigned int length;
	if (!inputBuf)
		return;

	length = strlen(inputBuf);
	if (length > LINELEN)
		length = LINELEN; // sanity check
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
				if (strcmp(entry, token) == 0) { // if this line's token is what we want, save it
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
	// get version info before init since we use system-serial-number (dell service tag) as a hook
	GenerateVersionInfo();

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

	checkEnvironmentForWebProxy();

	// remove old data
	int rc = unlink(TS_SERVERS_OLD);
	if (rc && errno != ENOENT)
		fprintf(stderr, "%s: unlink: %s\n", __FUNCTION__, strerror(errno));

	rc = pthread_create(&locationThreadId, NULL, locationThreadTask, NULL);
	if (rc)
		fprintf(stderr, "%s: pthread_create: %s\n", __FUNCTION__, strerror(rc));

	rc = pthread_detach(locationThreadId);
	if (rc)
		fprintf(stderr, "%s: pthread_detach: %s\n", __FUNCTION__, strerror(rc));
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

void sigint_handler(const int sig)
{
	printf("Got interrupt request (signal %d), will process after timeout expires.\n", sig);
	running = 0;
}

void sigusr_handler(const int sig)
{
	if (verbose) {
		verbose = 0;
		AeDRMSetLogLevel(AeLogWarning);
	}
	else {
		++verbose;
		AeDRMSetLogLevel(AeLogDebug);
	}

	printf("Got SIGUSR1 (signal %d), setting verbose flag to '%s'.\n",
			sig, verbose ? "on" : "off");
}

void setUpSignalHandling()
{
	int rc;
	struct sigaction act;

	rc = sigemptyset(&act.sa_mask);
	if (rc)
		fprintf(stderr, "%s: sigemptyset: %s\n", __FUNCTION__, strerror(errno));

	act.sa_handler = sigint_handler;
	act.sa_flags = SA_RESTART;

	rc = sigaction(SIGINT, &act, NULL);
	if (rc)
		fprintf(stderr, "%s: sigaction: %s\n", __FUNCTION__, strerror(errno));
	//rc = sigaction(SIGQUIT, &act, NULL);
	//if (rc)
	//	fprintf(stderr, "%s: sigaction: %s\n", __FUNCTION__, strerror(errno));
	rc = sigaction(SIGTERM, &act, NULL);
	if (rc)
		fprintf(stderr, "%s: sigaction: %s\n", __FUNCTION__, strerror(errno));

	act.sa_handler = sigusr_handler;
	act.sa_flags = SA_RESTART;
	rc = sigaction(SIGUSR1, &act, NULL);
	if (rc)
		fprintf(stderr, "%s: sigaction: %s\n", __FUNCTION__, strerror(errno));
}

int main(int argc, char *argv[])
{
	printf("RSM_TS Agent built " __DATE__ " " __TIME__ "\n");
	printf("OPENSSL Version: %s\n", OPENSSL_VERSION_TEXT);

	setUpSignalHandling();

	char *site = 0;

	// process cmd-line args
	int argcc = 1;
	while (argcc < argc) {
		if (argv[argcc][0] == '-') {
			switch (argv[argcc][1]) {
			case 'v': // verbose bump
				verbose++;
				if (strlen(argv[argcc]) >= 3 && argv[argcc][2] == 'v')
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
		printf("Using web proxy: host:%s port:%d user:%s\n", proxyInfo.pHost, proxyInfo.iPort, proxyInfo.pUser);
		rc = AeWebSetProxy(AeWebProxyProtoHTTP, proxyInfo.pHost, proxyInfo.iPort, proxyInfo.pUser, proxyInfo.pPass);
		if (rc == AeEOK)
			printf("Web proxy was set successfully.\n");
		else
			fprintf(stderr, "Failed to set proxy, connecting directly: %s\n", AeGetErrorString(rc));
	}
	else {
		if (verbose)
			printf("Not using web proxy.\n");
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

	// main loop - sends heartbeat at ping rate, and notifies of any alarm conditions

	printf("Ready\n");

	// Identifies the type of computer hardware the server is running on.
	// Dell T7500 or Dell T620 for example.
	// product_info.alt file is written during RSM_launch startup script execution
	SendHardwareName(&dataItem, "product_info.alt");
	// Records version of BIOS installed on server
	SendBiosVersion(&dataItem, "product_info.alt");

	running = 1;
	while (running) {
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

#if 0
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
#endif

int UpdateDataItem(StatusType status, AeDRMDataItem *dataItem)
{
	char cmd[CMD_LENGTH];
	int ret = 0;
	int rc = 0;
	int gpuFound = 0, allRevsValid = 0;
	static int prevGpuFound = 0, prevAllRevsValid = 0;

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
		ret = 0; // caller will post if this is non-zero, so always return 0; handle posts via callback here
		// eventLogCheck(EventPostCB, dataItem);
		break;

	case STATUS_VERSIONS: {
		// check to see if package database has been modified, if so we need to update our version info
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

	case STATUS_OS_VERSION: {
		// report Ubuntu version from /etc/lsb-release (10.04, 14.04, etc.)
		char OsVersion[OS_VERSION_SIZE] = {0};
		int rc = ReadOsVersion(OsVersion, OS_VERSION_SIZE);

		if (rc == 0 && strcmp(OsVersion, savedOsVersion) != 0) {
			snprintf(savedOsVersion, LINELEN, "%s", OsVersion);
			AeGetCurrentTime(&dataItem->value.timeStamp);
			dataItem->pName = "TS.Version.OS";
			dataItem->value.data.pString = savedOsVersion;
			dataItem->value.iType = AeDRMDataString;
			dataItem->value.iQuality = AeDRMDataGood;
			AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
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

	case STATUS_LOCATIONINFO: {
		// check to see if the location info has been modified and if so update Axeda
		struct stat buf;
		memset(&buf, 0, sizeof(buf));
		int retVal = stat(LOC_FILE, &buf);
		if (retVal == 0 && buf.st_mtime != lastLocationInfoTime) {
			lastLocationInfoTime = buf.st_mtime;
			SendLocationInfo(dataItem);
			// no need to set ret to 1, we have already posted the data
		}
	} break;

	case STATUS_SERVICES:
		SendServersStatus(dataItem);
		ret = 0; // caller will post data again if this is non-zero
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
		snprintf(cmd, CMD_LENGTH, "ls -tr1 %s/TSexperiment-*.txt > %s 2>/dev/null", SPOOL_DIR, TS_EXPERIMENTS);
		if (0 == system(cmd)) {
			SendExperimentMetrics(dataItem, TS_EXPERIMENTS);
		}
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

		agentInfo.serialNumberList = (char **)malloc(sizeof(char *)*agentInfo.numInsts); // list of string pointers

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
				// can specify each instr. by name, but 'Default' instr. can still be hooked into for SAP lookups
				dataItem->pName = "TS.PGM.Default";
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
		break;

	case STATUS_RAIDINFO:
		readAlarmsFromJson(RAIDINFO_FILE);
		break;

	case STATUS_GPU:
		readGpuErrorFile(gpuErrorFile, &gpuFound, &allRevsValid);
		if (verbose)
			printf("prevGpuFound %d prevAllRevsValid %d gpuFound %d allRevsValid %d\n",
					prevGpuFound, prevAllRevsValid, gpuFound, allRevsValid);
		if (gpuFound != prevGpuFound || allRevsValid != prevAllRevsValid) {

			dataItem->pName = "TS.GPU";
			dataItem->value.iQuality = AeDRMDataGood;
			dataItem->value.iType = AeDRMDataString;
			ret = 1;

			if (gpuFound && allRevsValid)
				dataItem->value.data.pString = "No problems.";
			else if (!gpuFound)
				dataItem->value.data.pString = "GPU not detected.";
			else if (!allRevsValid)
				dataItem->value.data.pString = "Lost connection to GPU.";

			prevGpuFound = gpuFound;
			prevAllRevsValid = allRevsValid;
		}
		break;
	}

	return ret;
}

void readGpuErrorFile(char const * const filename, int *gpuFound, int *allRevsValid)
{
	JSON_Value *root_value;
	JSON_Object *obj;

	// Read the gpuErrors file into memory.
	root_value = json_parse_file(filename);
	if (!root_value) {
		//printf("readGpuErrorFile root_value null\n");
		return;
	}

	obj = json_value_get_object(root_value);
	if (!obj) {
		json_value_free(root_value);
		//printf("readGpuErrorFile object null\n");
		return;
	}

	*gpuFound = json_object_get_boolean(obj, "gpuFound");
	*allRevsValid = json_object_get_boolean(obj, "allRevsValid");

	json_value_free(root_value);
}	// end readGpuErrorFile

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

// returns 0 for success or non-zero for failure (standard C convention for return values)
int ReadOsVersion(char *OsVersion, const size_t OsVersionSize)
{
	if (!OsVersion || OsVersionSize < 6)
		return 1; // fail, invalid inputs

	*OsVersion = '\0';
	int success = 1; // init to 1, meaning fail, os version not updated

	FILE *fp = fopen("/etc/lsb-release", "rb");
	if (fp) {
		char line[LINELEN];
		while (fgets(line, LINELEN, fp)) {
			if (strstr(line, "DISTRIB_RELEASE")) {
				char *equalsign = strchr(line, '=');
				if (equalsign) {
					trimTrailingWhitespace(line);
					snprintf(OsVersion, OsVersionSize, "%s", equalsign + 1);
					success = 0; // success, os version updated
					break;
				}
			}
		}
		fclose(fp);
	}

	return success;
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
					if (numFileSystems == 1) {
						fileSystemList = (FileSystemList *)malloc(sizeof(FileSystemList));
					}
					else {
						FileSystemList *tempFileSystemList = (FileSystemList *)realloc(fileSystemList,
								numFileSystems * sizeof(FileSystemList));
						if (tempFileSystemList)
							fileSystemList = tempFileSystemList;
						else
							fprintf(stderr, "%s: realloc: out of memory\n", __FUNCTION__);
					}
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
	GetSoftwareVersion("dbreports", "Version", dataItem, TS_VERSIONS);
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
}

static void SendLocationInfo(AeDRMDataItem *dataItem)
{
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

char *getTextUpToDelim(char * const start, const char delim, char * const output, const int outputSize)
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

// Read the proxy info directly from /etc/environment, instead of using getenv(),
// so that we get the latest settings.
// /etc/environment populates the environment at boot only.
// Any changes to /etc/environment are not visible in environment till next boot.
// We use the value of https_proxy if present
void checkEnvironmentForWebProxy()
{
	FILE *fp;
	char line[LINE_LENGTH];
	const char *token[] = {"https_proxy=http", "http_proxy=http"}; // order is important
	const int nTokens = sizeof(token) / sizeof(token[0]);

	// initialize web proxy struct
	proxyInfo.useProxy =   0;
	proxyInfo.pHost[0] = '\0';
	proxyInfo.iPort    =   0;
	proxyInfo.pUser[0] = '\0';
	proxyInfo.pPass[0] = '\0';

	fp = fopen("/etc/environment", "r");
	if (!fp)
		return;

	while (fgets(line, LINE_LENGTH, fp)) {
		trimTrailingWhitespace(line);

		// Remove comments (whole-line or suffixed) before we look at the text.
		char *commentChar = strchr(line, '#');
		if (commentChar)
			*commentChar = '\0';

		// Try to find either of the two tokens we accept.
		for (int ii = 0; ii < nTokens; ++ii) {
			// sample proxy lines:
			// https_proxy=https://user:pass@1.2.3.4:5
			// https_proxy=http://user:pass@1.2.3.4:5
			// http_proxy=https://user:pass@1.2.3.4:5
			// http_proxy=http://user:pass@1.2.3.4:5
			// https_proxy=https://1.2.3.4:5
			// https_proxy=http://1.2.3.4:5
			// http_proxy=https://1.2.3.4:5
			// http_proxy=http://1.2.3.4:5
			// #https_proxy=http://1.2.3.4:5
			// #http_proxy=http://1.2.3.4:5
			const int tlen = strlen(token[ii]);
			if (strstr(line, token[ii])) {
				char buf[LINE_LENGTH];
				char *pos = line + tlen;
				while (*pos++ != ':') // may be http, may be https; skip to next :
					;
				pos += 2; // skip the // characters

				if (strchr(line, '@')) { // user:pass@ may or may not be present
					pos = getTextUpToDelim(pos, ':', proxyInfo.pUser, LINE_LENGTH);
					pos = getTextUpToDelim(pos, '@', proxyInfo.pPass, LINE_LENGTH);
				}
				pos = getTextUpToDelim(pos, ':', proxyInfo.pHost, LINE_LENGTH);
				pos = getTextUpToDelim(pos, '\0', buf, LINE_LENGTH);

				proxyInfo.iPort = atoi(buf);
				proxyInfo.useProxy = 1;
				break; // done, don't check for the other case
			}
		}
	}
	fclose(fp);
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

// Server status queries

void getKeyAndValue(char *inputBuf, const char delim, char **key, char **value)
{
	char *cpos;
	const char space = 32;

	if ((!inputBuf) || (!key) || (!value))
		return;

	*key = NULL;
	*value = NULL;

	cpos = strchr(inputBuf, delim);
	if (cpos) {
		*key = inputBuf;
		*cpos = '\0';
		trimTrailingWhitespace(*key);

		cpos++;
		while (*cpos == space)
			cpos++;
		*value = cpos;
	}
}	// end getKeyAndValue

void *growBufIfNeeded(void *buf, size_t *currentSize, size_t sizeNeeded)
{
	if (!currentSize) // buf can be NULL
		return NULL;

	void *newPtr = NULL;;
	if (*currentSize < sizeNeeded)
		newPtr = realloc(buf, sizeNeeded);

	if (newPtr) {
		*currentSize = sizeNeeded;
		return newPtr;
	}
	else {
		return buf;
	}
}

void parseServerStatus(char const * const filename, serverStatus_t * const serverStatus)
{
	if (!filename || !serverStatus)
		return;

	serverStatus->count = 0;
	serverStatus->server = NULL;
	serverStatus->size = 0;

	// Preallocate a guess at what we'll need.
	serverStatus->server = growBufIfNeeded(
			serverStatus->server,
			&serverStatus->size,
			32 * sizeof(serverStatusItem_t));

	char line[LINELEN];
	FILE *fp = fopen(filename, "rb");
	if (fp) {
		while (fgets(line, LINELEN, fp)) {
			trimTrailingWhitespace(line);

			char *key;
			char *value;
			getKeyAndValue(line, '|', &key, &value);

			serverStatus->server = growBufIfNeeded(
					serverStatus->server,
					&serverStatus->size,
					(serverStatus->count + 1) * sizeof(serverStatusItem_t));

			if (serverStatus->server) {
				serverStatus->server[serverStatus->count].name = strdup(key);
				serverStatus->server[serverStatus->count].status = strdup(value);
				serverStatus->count++;
			}
		}

		fclose(fp);
	}
}

void serverStatusDtor(serverStatus_t * const serverStatus)
{
	if (!serverStatus)
		return;

	for (int ii = serverStatus->count - 1; ii >= 0; --ii) { // loop backward, indexes are always valid
		free(serverStatus->server[ii].name);
		serverStatus->server[ii].name = NULL;

		free(serverStatus->server[ii].status);
		serverStatus->server[ii].status = NULL;

		serverStatus->count--;
	}

	if (serverStatus->server)
		free(serverStatus->server);

	serverStatus->size = 0;
	serverStatus->server = NULL;
}

void SendTimeStamp(char const * const name1, char const * const name2)
{
	AeDRMDataItem dataItem;
	char strbuf[LINELEN] = {0};
	struct tm tm;
	time_t unixtime = time(NULL);

	AeGetCurrentTime(&dataItem.value.timeStamp);
	gmtime_r(&unixtime, &tm); // get GMT representation of system time

	snprintf(strbuf, LINELEN, "%04d-%02d-%02d %02d:%02d:%02d GMT",
			1900 + tm.tm_year, 1 + tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

	WriteAeStringDataItem(name1, name2, strbuf, &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
}

int checkForServiceStatusChange(
		serverStatus_t const * const serverStatusOld,
		serverStatus_t const * const serverStatusNew, const int ii)
{
	if (!serverStatusOld || !serverStatusNew)
		return -1;

	if (serverStatusOld->count != serverStatusNew->count)
		return 1;

	if (strncmp(serverStatusOld->server[ii].name,   serverStatusNew->server[ii].name,   LINE_LENGTH) != 0)
		return 1;

	if (strncmp(serverStatusOld->server[ii].status, serverStatusNew->server[ii].status, LINE_LENGTH) != 0)
		return 1;

	return 0;
}

void SendServersStatus(AeDRMDataItem *dataItem)
{
	serverStatus_t serverStatusOld;
	serverStatus_t serverStatusNew;

	AeGetCurrentTime(&dataItem->value.timeStamp);

	// Get the list of service statuses using the same code
	// that populates the /configure/services/ page on the TS
	char const * const cmd = "python " BIN_DIR "/status.py 1>/dev/null 2>" TS_SERVERS;
	int rc = system(cmd);
	if (rc == -1)
		printf("%s: system: %s\n", __FUNCTION__, strerror(errno));

	parseServerStatus(TS_SERVERS_OLD, &serverStatusOld);
	parseServerStatus(TS_SERVERS, &serverStatusNew);

	for (int ii = 0; ii < serverStatusNew.count; ++ii) {

		int statusChanged = checkForServiceStatusChange(&serverStatusOld, &serverStatusNew, ii);

		// If status of service has changed, send the data item for that service.
		if (statusChanged) {
			WriteAeStringDataItem("TS.Server",
					serverStatusNew.server[ii].name, serverStatusNew.server[ii].status, dataItem);
			AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, dataItem);
		}
	}

	// Always send the timestamp when the service status was checked.
	// Better to update one timestamp data item every time, than to update 15 data items, one per service.
	SendTimeStamp("TS.Server", "lastChecked");

	serverStatusDtor(&serverStatusNew);
	serverStatusDtor(&serverStatusOld);

	rc = rename(TS_SERVERS, TS_SERVERS_OLD);
	if (rc == -1)
		fprintf(stderr, "%s: rename: %s\n", __FUNCTION__, strerror(errno));
}

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

static int rsshConnection(char const * const rsshCmd,
						  char const * const remoteHost,
						  const int remoteConnectionPort,
						  char const * const user,
						  char const * const pass)
{
	int retval = 0; // init to failure case

	srand(time(NULL));
	int remoteTunnelPort = 15000 + (rand() % 1024);

	if (verbose)
		printf("%s: %s to %s:%d on port %d\n", __FUNCTION__,
				rsshCmd, remoteHost, remoteConnectionPort, remoteTunnelPort);

	ssh_tunnel_create(remoteHost, user, pass, remoteConnectionPort, remoteTunnelPort);

	// Create a file to upload to the Axeda Enterprise server
	FILE *fp = fopen(RSSH_CMD_FILE, "w");
	if (fp == NULL) {
		printf("%s: failed to open " RSSH_CMD_FILE "\n", __FUNCTION__);
		return 0; // fail
	}

	if (0 == strcmp(rsshCmd, "start")) {
		// Create a file containing the port info the caller needs to connect
		fprintf(fp, "Log in to %s and run:\r\n"
				"ssh -l ionadmin -p %d -o NoHostAuthenticationForLocalhost=yes -o StrictHostKeyChecking=no localhost\r\n",
				remoteHost, remoteTunnelPort);
	}
	else {
		fprintf(fp, "Dummy file to close reverse SSH session\r\n");
	}
	fclose(fp);

	return retval;
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
		if (strstr(pUpload->ppUploads[pUpload->iUploadIdx]->pName, "rssh-start") != 0) {
			// parse out the info from the 'command' file name
			// expected file name is:
			//    rssh-start-remoteHost-portOnRemoteHost-username-password
			char *rsshCmd;
			char *host;
			char *pstr;
			char *user;
			char *pass;
			char tokens[1024];
			snprintf(tokens, sizeof(tokens), "%s", pUpload->ppUploads[pUpload->iUploadIdx]->pName);
			strtok(tokens, "-"); /* we don't care about the /rssh- part */
			rsshCmd = strtok(NULL, "-");
			host = strtok(NULL, "-");
			pstr = strtok(NULL, "-");
			user = strtok(NULL, "-");
			pass = strtok(NULL, "-");

			int port;
			sscanf(pstr, "%d", &port);

			// execute the rssh command
			if (rsshCmd && host && pass && user) {
				rsshfile = 1;
				rsshConnection(rsshCmd, host, port, user, pass);
			}

			// Erase copy of file name.
			memset(tokens, 0, 1024);
		}
		else if (strstr(pUpload->ppUploads[pUpload->iUploadIdx]->pName, "rssh-stop") != 0) {
			rsshfile = 1;
			ssh_tunnel_remove();
		}

		// open file
		if (rsshfile == 1) {
			pUpload->iCurFileHandle = AeFileOpen(RSSH_CMD_FILE, AE_OPEN_READ_ONLY);
		}
		else {
			pUpload->iCurFileHandle = AeFileOpen(pUpload->ppUploads[pUpload->iUploadIdx]->pName, AE_OPEN_READ_ONLY);
		}
		if (pUpload->iCurFileHandle == AeFileInvalidHandle) {
			return AeFalse;
		}

		pUpload->curFileStat.pName = pUpload->ppUploads[pUpload->iUploadIdx]->pName;
		pUpload->curFileStat.iType = AeFileTypeRegular;
		if (rsshfile == 1) {
			pUpload->curFileStat.iSize =
#ifndef ENABLE_LARGEFILE64
					AeFileGetSize
#else
					AeFileGetSize64
#endif
					(RSSH_CMD_FILE);
		}
		else {
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
	if (*piSize < 0) {
		return AeFalse;
	}
	else if (*piSize == 0) {
		AeFileClose(pUpload->iCurFileHandle);
		pUpload->iCurFileHandle = AeFileInvalidHandle;

		if (pUpload->ppUploads[pUpload->iUploadIdx]->bDelete)
			AeFileDelete(pUpload->ppUploads[pUpload->iUploadIdx]->pName);

		char *filename = pUpload->ppUploads[pUpload->iUploadIdx]->pName;
		if (strstr(filename, "rssh-start") != 0) {
			// Erase the upload filename "rssh-start-username-password" from memory.
			memset(filename, 0, strlen(filename));
			// Erase the copy we made of the upload filename.
			memset(pUpload->curFileStat.pName, 0, strlen(pUpload->curFileStat.pName));
		}

		pUpload->iUploadIdx += 1;
	}
	else if (*piSize > 0) {
		*ppData = pUpload->pBuffer;
	}

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

void *locationThreadTask(void *args __attribute__((unused)))
{
	int rc = system(BIN_DIR "/location_helper.sh");
	if (rc == -1)
		printf("%s: system: %s\n", __FUNCTION__, strerror(errno));
	locationThreadExited = 1;
	return NULL;
}
