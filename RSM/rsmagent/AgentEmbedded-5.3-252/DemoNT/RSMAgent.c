//
// Remote Support and Monitor Agent
// (c) 2011 Life Technologies, Ion Torrent
//


#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <string.h>

#include "AeOSLocal.h"
#include "AeTypes.h"
#include "AeError.h"
#include "AeOS.h"
#include "AeInterface.h"

#include "RSMLogger.h"

#define DEFAULT_OWNER "drm-data_source"
#define PGM_CONFIG "/software/gui/cntrl/Controller.config"
#define PGM_VERSIONS "/software/gui/cntrl/PGM_Versions.txt"

typedef enum {
	PGM_STATUS_TASK = 0,
	PGM_STATUS_HD,
	PGM_STATUS_TEMP,
	PGM_STATUS_PRES,
	PGM_STATUS_NUMITEMS
} PGMStatusType;

typedef struct {
	char	*modelNumber;
	char	*serialNumber;
	int	pingRate;
} AgentInfo;

// globals
AgentInfo agentInfo;
int ok = 1;
int verbose = 0;


int UpdateDataItem(PGMStatusType status, AeDRMDataItem *dataItem);
int UpdateAlarmItem(PGMStatusType status, AeDRMAlarm *alarmItem, AeDRMDataItem *dataItem);

// GetConfigEntry - returns the requested entry into buf and returns 0 or error code
int GetConfigEntry(char *configFile, char delimiter, char *entry, char *buf, int bufSize)
{
	int ret = 1; // entry not found by default

	FILE *fp = fopen(configFile, "r");
	if (fp) {
		char line[1024];
		char *ptr;
		while (fgets(line, sizeof(line), fp)) {
			ptr = strchr(line, '\n'); if (ptr) *ptr = 0;
			ptr = strchr(line, '\r'); if (ptr) *ptr = 0;
			ptr = strchr(line, delimiter);
			if (ptr) {
				*ptr = 0;
				ptr++;
				if (strcmp(entry, line) == 0) {
					strncpy(buf, ptr, bufSize);
				}
				buf[bufSize-1] = 0; // in case ptr string exceeded buf len
				ret = 0;
			}
		}
	}

	if (verbose > 0) {
		if (ret == 0)
			printf("Entry: %s Value: %s\n", entry, buf);
		else
			printf("Entry: %s not found?\n", entry);
	}

	return ret;
}

void RSMInit()
{
	// initialize the AgentInfo fields
	agentInfo.modelNumber = strdup("PGM");

	char buf[256];
	int ret = GetConfigEntry(PGM_CONFIG, ':', "Serial", buf, sizeof(buf));
	if (ret == 0)
		agentInfo.serialNumber = strdup(buf);
	else
		agentInfo.serialNumber = strdup("unknown");
	agentInfo.pingRate = 30;
}

void RSMClose()
{
	free(agentInfo.modelNumber);
	free(agentInfo.serialNumber);
}

void GetSoftwareVersion(char *softwareComponent, AeDRMDataItem *item)
{
	static char buf[256];
	buf[0] = 0;

	item->pName = softwareComponent;
	item->value.data.pString = buf;
	item->value.iType = AeDRMDataString;
	item->value.iQuality = AeDRMDataGood;
	// some components are handled differently, most come from our versions file
	if (strcmp(softwareComponent, "TYPE") == 0) {
		strcpy(buf, "PGM");
	} else {
		int ret = GetConfigEntry(PGM_VERSIONS, '-', softwareComponent, buf, sizeof(buf));
		if (ret > 0)
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
	printf("RSM Agent\n");

	char *site = 0;

	// process cmd-line args
	int argcc = 1;
	while (argcc < argc) {
		if (argv[argcc][0] == '-') {
			switch (argv[argcc][1]) {
				case 'v':
					verbose++;
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

	AeInt32 iDeviceId, iServerId;
	AeTimeValue pingRate, timeLimit;
	AeError rc;

	// initialize the Axeda embedded agent
	AeInitialize();

	// set up a few options
#ifdef HAVE_OPENSSL
	rc = AeWebSetSSL(AeWebCryptoMedium, AeFalse, NULL);
	if (rc != AeEOK) {
		fprintf(stderr, "Failed to set SSL parameters (%s)\n", AeGetErrorString(rc));
		return 1;
	}
#endif /* HAVE_OPENSSL */

	// configure master device
	rc = AeDRMAddDevice(AeDRMDeviceMaster, agentInfo.modelNumber, agentInfo.serialNumber, &iDeviceId);
	if (rc != AeEOK) {
		fprintf(stderr, "Failed to add device (%s)\n", AeGetErrorString(rc));
		return 1;
	}

	// Axeda debug output on
	if (verbose > 0)
		AeDRMSetLogLevel(AeLogDebug);

	// configure primary DRM server
	pingRate.iSec = agentInfo.pingRate;
	pingRate.iMicroSec = 0;
	rc = AeDRMAddServer(AeDRMServerConfigPrimary, site, DEFAULT_OWNER, &pingRate, &iServerId);
	if (rc != AeEOK) {
		fprintf(stderr, "Failed to add server (%s)\n", AeGetErrorString(rc));
		return 1;
	}

	printf("Initialized!\n");

	// send the static software version tags
	AeDRMDataItem dataItem;
	AeGetCurrentTime(&dataItem.value.timeStamp);
	GetSoftwareVersion("TYPE", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("Datacollect", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("LiveView", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("driver", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("fpga", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("Scripts", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("OS", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("Graphics", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("Board", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
	GetSoftwareVersion("Board_Serial", &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);

	// install interrupt handler
	signal(SIGINT, sigint_handler);

	// main loop - sends heartbeat at ping rate, and notifies of any alarm conditions

	printf("Ready\n");

	ok = 1;
	while (ok) {
		// here we can send any data updates we want, at the ping rate, such as PGM status (running, idle, etc)
		unsigned int i;
		for(i=0;i<PGM_STATUS_NUMITEMS;i++) {
			if (UpdateDataItem((PGMStatusType)i, &dataItem) > 0) {
				AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
				// here we can test on status items and see if they require an alarm be sent
printf("Item: %s Value: ", dataItem.pName);
if (dataItem.value.iType == AeDRMDataAnalog)
	printf("%.4lf\n", dataItem.value.data.dAnalog);
else if (dataItem.value.iType == AeDRMDataString)
	printf("%s\n", dataItem.value.data.pString);
else
	printf("?\n");

				AeDRMAlarm alarmData;
				if (UpdateAlarmItem((PGMStatusType)i, &alarmData, &dataItem) > 0) {
					AeDRMPostAlarm(iDeviceId, iServerId, AeDRMQueuePriorityUrgent, &alarmData);
printf ("ALARM!\n");
				}
			}
		}

		// set time limit for the DRM execution. this is also the data poll rate.
		timeLimit.iSec = 30;
		timeLimit.iMicroSec = 0;
		AeDRMExecute(&timeLimit);
	}

	//  shutdown Axeda Agent Embedded 
	printf("Shutting down...\n");
	AeShutdown();
	RSMClose();

	printf("Done.\n");
	return 0;
}

int UpdateDataItem(PGMStatusType status, AeDRMDataItem *dataItem)
{
	int ret = 0;

	AeGetCurrentTime(&dataItem->value.timeStamp);

	switch (status) {
		case PGM_STATUS_TASK:
			dataItem->pName = "Task";
			dataItem->value.iType = AeDRMDataString;
			dataItem->value.iQuality = AeDRMDataGood;
			dataItem->value.data.pString = "Idle"; // TODO replace with PGM task: Idle, Pre beadfind, Post Beadfind, Seq, Clean
			ret = 1;
		break;

		case PGM_STATUS_HD:
			dataItem->pName = "HD";
			dataItem->value.iType = AeDRMDataAnalog;
			dataItem->value.iQuality = AeDRMDataGood;
			dataItem->value.data.dAnalog = 0.6; // TODO fill in with results drive percent full
			ret = 1;
		break;

		case PGM_STATUS_TEMP:
			dataItem->pName = "TEMP";
			dataItem->value.iType = AeDRMDataAnalog;
			dataItem->value.iQuality = AeDRMDataGood;
			dataItem->value.data.dAnalog = 37.1; // TODO fill in with actual temperature
			ret = 1;
		break;
	}

	return ret;
}
 
int UpdateAlarmItem(PGMStatusType status, AeDRMAlarm *alarmItem, AeDRMDataItem *dataItem)
{
	int ret = 0;

	alarmItem->timeStamp = dataItem->value.timeStamp;
	alarmItem->bAck = AeFalse;
	alarmItem->bActive = AeTrue;
	alarmItem->pDataItem = dataItem;

	switch (status) {
		case PGM_STATUS_TEMP:
			if (dataItem->value.data.dAnalog < 25.0) { // low temperature
				alarmItem->iSeverity = 30;
				alarmItem->pCondition = "Low";
				alarmItem->pDescription = "Low Temperature";
				ret = 1;
			} else if (dataItem->value.data.dAnalog > 35.0) { // high temperature
				alarmItem->iSeverity = 30;
				alarmItem->pCondition = "High";
				alarmItem->pDescription = "High Temperature";
				ret = 1;
			}
		break;
	}

	return ret;
}
 
