#include <stdio.h>
#include <string.h>

#include "AlarmMgr.h"
#include "RSMAgent.h"
#include "AeInterface.h"

#define MAX_ALARMS 32
#define LINELEN 256
#define ALARM_CACHE "/var/spool/ion/RsmAlarmCache"

enum LogEntryType { SETTING_ALARM, CLEARING_ALARM, EVENT };

unsigned int numStandingAlarms = 0;
Alarms_t StandingAlarms[MAX_ALARMS];
static AeInt32 iDeviceId = 0;
static AeInt32 iServerId = 0;

static void sendAlarmByName(enum LogEntryType action, char const * const name, char const * const message);
//static void GetNameFromEventLogEntryDescription(char * name, enum LogEntryType logEntryType, const char *entry);

#ifdef UNITTEST
int main(void)
{
	AlarmMgrInit();
	AlarmMgr_AddAlarm("Alarm Type 1");
	AlarmMgr_AddAlarmByName("Alarm.Type2", "Specific data explaining Alarm Type 2");
	AlarmMgr_DelAlarm("Alarm Type 1");
	AlarmMgr_DelAlarmByName("Alarm.Type2");
	AlarmMgrSave();

	return 0;
}
#endif

void AlarmMgrInit(AeInt32 iDeviceIdIn, AeInt32 iServerIdIn)
{
	iDeviceId = iDeviceIdIn;
	iServerId = iServerIdIn;
	memset(StandingAlarms, 0, sizeof(StandingAlarms));
	// Read standing alarms from disk file.
	// We need to carry alarms across restarts, or else we won't
	// ever clear an alarm for PCIE wrong slot, since that changes
	// only when power is off.  We don't clear an alarm if it doesn't appear
	// in StandingAlarms, so we must populate StandingAlarms at start-up.
	FILE *fp = fopen(ALARM_CACHE, "r");
	unsigned int ii = 0;
	char line[LINELEN];
	if (fp) {
		while (fgets(line, LINELEN, fp)) {
			trimTrailingWhitespace(line);
			int items = sscanf(line, "startTime:%ld endTime:%ld active:%d",
					&StandingAlarms[ii].start_t,
					&StandingAlarms[ii].end_t,
					&StandingAlarms[ii].active);
			if (items != 3) {
				printf("%s: Unexpected format in %s\n", __FUNCTION__, ALARM_CACHE);
				continue;
			}
			char *msgPos = strstr(line, " msg:");
			if (msgPos) {
				*msgPos = '\0'; // null-terminate the name string that is before the msg string
				msgPos += 5;    // skip past " msg:" (note leading space, it's in the string and must be considered)
				snprintf(StandingAlarms[ii].msg, 252, "%s", msgPos);
			}
			char *namePos = strstr(line, "name:");
			if (namePos) {
				namePos += 5;   // skip past "name:"
				snprintf(StandingAlarms[ii].name, 256, "%s", namePos);
			}
			++ii;
		}
		fclose(fp);
		numStandingAlarms = ii;
	}
	else {
		printf("Failed to open alarm cache file %s for read\n", ALARM_CACHE);
	}
}

void AlarmMgrSave(void)
{
	// write standing alarms to disk file
	FILE *fp = fopen(ALARM_CACHE, "w");
	if (fp) {
		int ii;
		for (ii = 0; ii < MAX_ALARMS; ++ii) {
			if (StandingAlarms[ii].active)
				fprintf(fp, "startTime:%ld endTime:%ld active:%d name:%s msg:%s\n",
						StandingAlarms[ii].start_t,
						StandingAlarms[ii].end_t,
						StandingAlarms[ii].active,
						StandingAlarms[ii].name,
						StandingAlarms[ii].msg);
		}
		fclose(fp);
	}
	else {
		printf("Failed to open alarm cache file %s for write\n", ALARM_CACHE);
	}
}

void AlarmMgr_AddAlarmByName(char const * const AlarmName, // for example, Alarm.DiskError
		char const * const AlarmTxt) // arbitrary details
{
	int i;
	for (i = 0; i < MAX_ALARMS; i++) {
		if ((StandingAlarms[i].active) && !strcmp(StandingAlarms[i].name, AlarmName)) {
			// this alarm already exists
			break;
		}
	}
	if (i == MAX_ALARMS) {
		for (i = 0; i < MAX_ALARMS; i++) {
			// look for an open slot
			if (!StandingAlarms[i].active) {
				strncpy(StandingAlarms[i].msg,  AlarmTxt,  sizeof(StandingAlarms[i].msg ));
				strncpy(StandingAlarms[i].name, AlarmName, sizeof(StandingAlarms[i].name));
				StandingAlarms[i].start_t = time(NULL);		//record the alarm start time
				StandingAlarms[i].active = 1;
				numStandingAlarms++;
				sendAlarmByName(SETTING_ALARM, AlarmName, AlarmTxt);
				break;
			}
		}
	}
}

void AlarmMgr_DelAlarmByName(char const * const AlarmName)
{
	int i;
	for (i = 0; i < MAX_ALARMS; i++) {
		if (StandingAlarms[i].active && !strcmp(StandingAlarms[i].name, AlarmName))	{
			sendAlarmByName(CLEARING_ALARM, AlarmName, StandingAlarms[i].msg);
			memset(&StandingAlarms[i], 0, sizeof(StandingAlarms[i]));
			numStandingAlarms--;
			break;
		}
	}
}

static void sendAlarmByName(enum LogEntryType action, char const * const name, char const * const message)
{
	char msg2[LINELEN];
	char *msgPtr = (char *)message;
	AeDRMDataItem dataItem;
	memset(&dataItem, 0, sizeof(AeDRMDataItem));
	AeGetCurrentTime(&dataItem.value.timeStamp);

	if (action == CLEARING_ALARM) {
		snprintf(msg2, LINELEN, "%s - CLEARED", message);
		msgPtr = msg2;
	}

	dataItem.pName = (char *)name;
	dataItem.value.data.pString = msgPtr;
	dataItem.value.iType = AeDRMDataString;
	dataItem.value.iQuality = AeDRMDataGood;
	UpdateDataItem(AeDRMQueuePriorityNormal, &dataItem);
	AeDRMPostDataItem(iDeviceId, iServerId, AeDRMQueuePriorityNormal, &dataItem);
}

/*
// converts description portion of log entry to camel case with only a-z and A-Z characters
static void GetNameFromEventLogEntryDescription(char * name, enum LogEntryType logEntryType, const char *entry)
{
	if (logEntryType == EVENT) {
		strcpy(name, "Event.");
	}
	else {
		strcpy(name, "Alarm.");
	}
	name += strlen(name);

	int capitalize = 0;
	while (*entry != '\0') {
		char c = *entry++;
		if (((c < 'A') || (c > 'z')) ||
			((c < 'a') && (c > 'Z'))    ) { // not a-z or A-Z

			// end event name on character other than space between words
			if (c != ' ')
				break;

			capitalize = 1;
			continue;
		}

		if (capitalize && ((c >= 'a') && (c <= 'z')))	{
			c = c - 0x20;
		}
		*name++ = c;
		capitalize = 0;
	}
	*name = '\0'; //NULL Terminate
}
*/
