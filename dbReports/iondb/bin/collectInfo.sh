#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
#	---------------------------------------------------------------------
#	Functions
#	---------------------------------------------------------------------
# Checks if an ethernet port exists
function valid_port ()
{
	grep -q $1 /proc/net/dev
	valid=$?
	return $valid
}
#	---------------------------------------------------------------------
#	Collect Torrent Server configuration information
#	---------------------------------------------------------------------
echo "=================================================================="
echo "Date Collected:"
date
echo
echo "=================================================================="
echo "Server Uptime:"
uptime
echo
echo "=================================================================="
echo "Ion Software Package Status:"
dpkg -l 'ion-*' tmap
echo
echo "=================================================================="
echo "Reference Library Installed:"
resultdir='/results/referenceLibrary'
tmapdir='tmap-f*/'
if [ -d $resultdir ]; then

    ls -1 $resultdir/$tmapdir/
    
else
	echo "No reference library"
fi
echo
echo "=================================================================="
echo "Serial Number:"
if [ -r /etc/torrentserver/tsconf.conf ]; then
	echo $(grep "serialnumber" /etc/torrentserver/tsconf.conf 2>/dev/null|cut -f2 -d":")
else
	echo "Not available"
fi
echo
echo "=================================================================="
echo "Hostname:" 
hostname --long
echo
echo "=================================================================="
echo "IP Addresses:"
hostname --all-ip-address
echo
echo "=================================================================="
echo "Network Configuration:"
echo
interfaces=( 0 1 2 3 4 )
for i in ${interfaces[@]}; do
	if (valid_port eth${i}); then
		/sbin/ifconfig eth${i}|grep 'inet addr'; fi
done
echo
echo "=================================================================="
echo "MAC Addresses:"
for i in ${interfaces[@]}; do
	if (valid_port eth${i}); then
		echo "eth${i}: " `/sbin/ifconfig eth${i}|grep HWaddr|awk '{print $5}'`; fi
done
echo
echo "=================================================================="
echo "SGE Queue Information:"
if (which qstat >/dev/null); then qstat -f; else echo "Could not find qstat"; fi
echo
echo "=================================================================="
echo "SGE Execute Hosts:"
if (which qconf >/dev/null); then qconf -sel; else echo "Could not find qconf";  fi
echo
echo "=================================================================="
echo "SGE Submit Hosts:"
if (which qconf >/dev/null); then qconf -ss; else echo "Could not find qconf";   fi
echo
echo "=================================================================="
echo "SGE Parallel Environments:"
if (which qconf >/dev/null); then qconf -spl; else echo "Could not find qconf";   fi
echo
echo "=================================================================="
echo "Exported directories:"
grep -v "#" /etc/exports
echo
echo "=================================================================="
echo "IPTABLES: (/etc/iptables.rules)"
if [ -r /etc/iptables.rules ];then
	cat /etc/iptables.rules
else
	echo "No such file"
fi
echo
echo "=================================================================="
echo "DNS/DHCP:"
echo "/etc/hosts file:"
cat /etc/hosts
echo
echo "/etc/dhcp3/dhcpd.conf file:"
if [ -r /etc/dhcp3/dhcpd.conf ];then
	cat /etc/dhcp3/dhcpd.conf
else
	echo "No such file"
fi
echo
echo "=================================================================="
echo "Required processes"
echo "PYTHON DAEMONS"
service ionCrawler status
service ionJobServer status
service ionArchive status
service ionPlugin status
service celeryd status
echo
echo "SGE DAEMONS"
ps aux|grep sge|grep -v grep
echo
echo "WEBSERVER"
ps aux|grep apache|grep -v grep
echo
echo "DATABASE"
ps aux|grep postgresql|grep -v grep

#	Disk Space Check
echo
echo "=================================================================="
echo "Disk Space"

#find space, echo Stale Mount if df fails
/opt/ion/iondb/bin/timeout.sh 5 df -PTh || echo "Error: The command df took longer than 5 seconds"

## N.B.  Needs superuser permission to execute!  won't work
##   Filesystem check interval
#echo
#echo "=================================================================="
#echo "Filesystem Check Interval"
#
#/sbin/tune2fs -l /dev/sda1 2>&1

#   Startup Scripts
echo
echo "=================================================================="
echo "Startup Scripts"

daemons=(
	postgresql
    nfs-kernel-server
    postfix
    sgeexec
    sgemaster
    ntp
    dhcp3-server
    apache2
    tomcat
    ionArchive
    ionCrawler
    ionJobServer
    ionPlugin
    celeryd
    RSM_Launch
    )
    
for d in ${daemons[@]}; do
	echo "Checking ${d}:"
	find /etc/rc* -name \*${d}\*
done

#	zombie Processes
echo
echo "=================================================================="
echo "Zombie Processes"
ps aux|grep defunct|grep -v grep

#	Network Time Protocol
echo
echo "=================================================================="
echo "Network Time Protocol"
echo
/usr/bin/ntpq -p

echo
echo "=================================================================="
echo "Database Information"
echo
/opt/ion/iondb/bin/db_check.sh

exit 0

