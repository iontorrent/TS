#!/bin/bash
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

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

# point to drmdev axeda server if on intranet
if [ $on_intranet = "Yes" ]; then
	while [ 1 ]; do
		/opt/ion/RSM/RSMAgent_TS https://drmdev.appliedbiosystems.com/eMessage >> /var/log/ion/RSMAgent_TS.log 2>&1
		echo "RSMAgent_TS exited with exit code $?.  Restarting in 10 seconds..." >&2
		sleep 10 # runaway protection
	done
else
	while [ 1 ]; do
		/opt/ion/RSM/RSMAgent_TS https://drm.appliedbiosystems.com/eMessage >> /var/log/ion/RSMAgent_TS.log 2>&1
		echo "RSMAgent_TS exited with exit code $?.  Restarting in 10 seconds..." >&2
		sleep 10 # runaway protection
	done
fi
