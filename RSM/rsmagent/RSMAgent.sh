#!/bin/sh

# Which Axeda server should we connect to?
# See if we can reach LifeTech internal DNS.
# Try multiple queries and see if any succeed.

on_intranet="No"

host updates.itw
if [ "$?" = "0" ]; then
	on_intranet="Yes"
fi

if [ $on_intranet != "Yes" ]; then
	host updates.ite
	if [ "$?" = "0" ]; then
		on_intranet="Yes"
	fi
fi

if [ $on_intranet != "Yes" ]; then
	host blackbird.itw
	if [ "$?" = "0" ]; then
		on_intranet="Yes"
	fi
fi

if [ $on_intranet != "Yes" ]; then
	host blackbird.ite
	if [ "$?" = "0" ]; then
		on_intranet="Yes"
	fi
fi

# point to drmdev axeda server if on lifetech LAN
if [ $on_intranet = "Yes" ]; then
	/opt/ion/RSM/RSMAgent_TS https://drmdev.appliedbiosystems.com/eMessage >> /var/log/RSMAgent_TS.log &
else
	/opt/ion/RSM/RSMAgent_TS https://drm.appliedbiosystems.com/eMessage >> /var/log/RSMAgent_TS.log &
fi
