#!/bin/sh

# point to drmdev axeda server if on lifetech LAN
host lifelink.corp.life
if [ "$?" = "0" ]
then
	/opt/ion/RSM/RSMAgent_TS https://drmdev.appliedbiosystems.com/eMessage >> /var/log/RSMAgent_TS.log &
else
	/opt/ion/RSM/RSMAgent_TS https://drm.appliedbiosystems.com/eMessage >> /var/log/RSMAgent_TS.log &
fi

