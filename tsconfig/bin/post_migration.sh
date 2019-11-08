#!/bin/bash
# Copyright (C) 2019 Thermo Fisher Scientific, Inc. All Rights Reserved
#
# Purpose: reset SGE hostname, and restore database and database migration
# 
# Usage: post_migration.sh postgresql_osupdate.backup
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

echo "*******************************************************************************"
echo "Restore database."
echo "*******************************************************************************"

# copy the backup file to the server and decompress it
databasefile=/results/postgresql_osupdate.backup 
if [[ -f /results/postgresql_osupdate.backup ]]; then
	# stop the Torrent Server background processes
	sudo /etc/init.d/ionCrawler stop
	sudo /etc/init.d/ionJobServer stop
	sudo /etc/init.d/ionPlugin stop
	sudo service celeryd stop
 
	# restart the service to clear database connections
	sudo /etc/init.d/postgresql restart
 
	# drop the existing iondb database
	sudo su - postgres -c "dropdb iondb"
 
	# create a new empty database
	sudo su - postgres -c "psql -q -c 'CREATE DATABASE iondb;'"
	sudo su - postgres -c "psql -q -c 'GRANT ALL PRIVILEGES ON DATABASE iondb to ion;'"
 
	# import data
	sudo su postgres -c "psql -q -e iondb < $databasefile"
 
	# start the Torrent Server background processes
	sudo /opt/ion/iondb/bin/check_system_services.py -s

	# 
	DJANGOADMIN=/opt/ion/manage.py

	cd /opt/ion/iondb
	# Legacy DB migration - Updates an existing database schema to 2.2
	/usr/bin/python ./bin/migrate.py

	# Django South DB migration - applies changes in rundb/migrations/
	/usr/bin/python ${DJANGOADMIN} migrate --all --ignore-ghost-migrations --noinput

	# Full syncdb to create ContentTypes and Permissions
	/usr/bin/python ${DJANGOADMIN} syncdb --noinput --all

	# Backfill tastypie API keys
	/usr/bin/python ${DJANGOADMIN} backfill_api_keys

	# Loads Test Fragment entries into database
	#/usr/bin/python ${DJANGOADMIN} loaddata template_init.json
	/usr/bin/python /opt/ion/iondb/bin/install_testfragments.py /opt/ion/iondb/rundb/fixtures/template_init.json

	# Collect static files (eg. admin pages css/js).
	# TS-9967 Purging content with clear instead of keeping old md5 hashed versions
	/usr/bin/python ${DJANGOADMIN} collectstatic --verbosity=0 --noinput --link --clear

	# Creates default report templates with site-specific content
	/usr/bin/python /opt/ion/iondb/bin/base_template.py
	/bin/chown www-data:www-data /opt/ion/iondb/templates/rundb/php_base.html

	# Updates existing database entries
	# THIS HAS TO HAPPEN AFTER THE PHP_BASE.HTML FILE IS PUT INTO PLACE DO NOT MOVE
	/usr/bin/python /opt/ion/iondb/bin/install_script.py

	# Loads system templates to database if needed
	echo "Going to add or update system plan templates if needed..."
	/usr/bin/python /opt/ion/iondb/bin/add_or_update_systemPlanTemplates.py
	
	# Loads DMFileSet entries into database
	/usr/bin/python /opt/ion/iondb/bin/install_dmfilesets.py

	cd -
else
    echo "$databasefile does not exist, please find the databasefile in results"
fi

echo "*****************************************************************************"
echo "Reset SGE queue hostname"
echo "*****************************************************************************"
newHostName=$(awk '{print $1}' /etc/hostname)
oldFQDN="ion-server" # Modify queues

for queue in $(qconf -sql); do
   qconf -sq $queue > /tmp/$queue.conf                 # dump queue
   sed -i "s/$oldFQDN/$newHostName/g" /tmp/$queue.conf   # modify queue
   qconf -Mq /tmp/$queue.conf                          # save queue
done
# Add submit host
qconf -as $newHostName

# Remove original host from gridengine host lists
if [[ "$oldFQDN" != "$newHostName" ]]; then
    (
        set +e
        qconf -de $oldFQDN
        qconf -ds $oldFQDN
        qconf -dh $oldFQDN

    )
fi


