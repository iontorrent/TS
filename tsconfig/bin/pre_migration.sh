#!/bin/bash
# Copyright (C) 2019 Thermo Fisher Scientific, Inc. All Rights Reserved
#
# Purpose: save database data 
#
#
set -e
#---------------------------------------
# Test for root
#---------------------------------------
if [ $(id -u) != 0 ]; then
    echo
    echo "Please run this script with root permissions:"
    echo
    echo "sudo $0"
    echo
    exit 1
fi

function backup_database()
{
	sudo pg_dump -U ion -c iondb > /results/postgresql_osupdate.backup
	sudo chown postgres:postgres /results/postgresql_osupdate.backup
	sudo chmod 644 /results/postgresql_osupdate.backup
	echo "postgresql_osupdate.backup has been generated in /results"
}

if [[ -f /results/postgresql_osupdate.backup ]]; then
    echo "There is one copy of postgresql_osupdate.backup in /results"
    exit 1
else
    echo "postgresql_osupdate.backup is not in /results!!"
    echo "Doing database backup now..."
    backup_database
fi    

