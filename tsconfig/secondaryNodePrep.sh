#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

set -u

############################################
#--- Include function definition file	---#
############################################
TSCONFIG_SRC_DIR='/usr/share/ion-tsconfig'
source $TSCONFIG_SRC_DIR/ts_params
source $TSCONFIG_SRC_DIR/ts_functions

[ $# -eq 0 ] && { echo "Usage: $0 <primary's hostname>" ; exit 1; }

#---------------------------------------#
#	Variables
#---------------------------------------#
DJANGO_SETTINGS=/opt/ion/iondb/settings.py
#TODO: Get hostname of primary TS
HOSTNAME_PRIMARY=$1
HOSTNAME_SECONDARY=$(hostname -s)

#---------------------------------------#
#	Edit settings.py
#---------------------------------------#
if [ -w $DJANGO_SETTINGS ]; then
	# DATABASE_HOST
	sed -i "s/DATABASE_HOST.*/DATABASE_HOST = \"$HOSTNAME_PRIMARY\"/" $DJANGO_SETTINGS
	# DATABASE_PORT
	sed -i "s/DATABASE_PORT.*/DATABASE_PORT = \"5432\"/" $DJANGO_SETTINGS
	
	#Make sure older settings.py files contain new variables
	grep -q 'SGEQUEUENAME' $DJANGO_SETTINGS
	[ $? -eq 1 ] && { echo "SGEQUEUENAME=" >> $DJANGO_SETTINGS ; }
	grep -q 'QMASTERHOST' $DJANGO_SETTINGS
	[ $? -eq 1 ] && { echo "QMASTERHOST=" >> $DJANGO_SETTINGS ; }
	
	# SGEQUEUENAME
	sed -i "s/SGEQUEUENAME.*/SGEQUEUENAME = \"$HOSTNAME_SECONDARY.q\"/" $DJANGO_SETTINGS
	# QMASTERHOST
	sed -i "s/QMASTERHOST.*/QMASTERHOST = \"$HOSTNAME_PRIMARY\"/" $DJANGO_SETTINGS
	
	echo "$DJANGO_SETTINGS file modified:"
	egrep '(DATABASE_HOST|DATABASE_PORT|SGEQUEUENAME|QMASTERHOST)' $DJANGO_SETTINGS
	echo
else
	echo $DJANGO_SETTINGS "cannot be edited for some reason."
	echo "Was this script run with sudo privileges?"
	echo "Is the ion-dbreports deb package installed?"
	exit 1
fi

#---	SGE		---#
#------------------------------------------------------------------------#
#	Uninstall SGE (it may have been installed as primary node by default)
#	2 cases: this is SGQ qmaster or this is execute host only.
#------------------------------------------------------------------------#
remove_sge

#---
#	Existing local /results needs to be renamed to /pgmdata
#	Then mount /results and $SGE_ROOT/$SGE_CELL from primary TS
#---
config_remote_dir $HOSTNAME_PRIMARY

#------------------------------#
#	Install SGE as compute node
#------------------------------#
echo -e "\nThis node needs to be added to the existing SGE configuration as an Admin Host\n"
echo -e "At the prompt, enter the password for the ionadmin user on the qmaster host\n"
ssh -t -l ionadmin $HOSTNAME_PRIMARY ". $SGE_ROOT/$SGE_CELL/common/settings.sh && qconf -ah $(hostname -f)"
config_compute_sge

#------------------------------#
#	ftp link in /home/ionguest
#------------------------------#
rm -f /home/ionguest/results
ln -sf /pgmdata /home/ionguest/results
echo "Modified links in ftp home"

#---
#	Disable exporting /results and $SGE_ROOT/$SGE_CELL
#---
exportfs -ua
sed -i '/\/results/d' /etc/exports
sed -i '/$SGE_ROOT/d' /etc/exports

#---	Torrent Server Daemons	---#
#----------------------------------------------------#
#	Shutdown server if its currently running
#	Prevent these daemons from starting at boot
#----------------------------------------------------#
invoke-rc.d ionJobServer stop
update-rc.d -f ionJobServer remove
update-rc.d ionJobServer stop 97 2 3 4 5 .

invoke-rc.d ionArchive stop
update-rc.d -f ionArchive remove
update-rc.d ionArchive stop 97 2 3 4 5 .

invoke-rc.d ionCrawler stop
update-rc.d -f ionCrawler remove
update-rc.d ionCrawler stop 97 2 3 4 5 .

psqldaemon=$(basename $(find /etc/init.d -name postgres\*))
invoke-rc.d $psqldaemon stop
update-rc.d -f $psqldaemon remove
update-rc.d $psqldaemon stop 19 2 3 4 5 .

invoke-rc.d apache2 stop
update-rc.d -f apache2 remove
update-rc.d apache2 stop 09 2 3 4 5 .

exit 0
