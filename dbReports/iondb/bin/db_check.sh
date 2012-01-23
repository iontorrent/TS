#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

ZERO_EXPECTED_LIST=(
auth_group
auth_group_permissions
auth_message
auth_user_groups
auth_user_user_permissions
django_admin_log
django_session
rundb_analysismetrics
rundb_backup
rundb_backupconfig
rundb_cruncher
rundb_emailaddress
rundb_experiment
rundb_libmetrics
rundb_plugin
rundb_referencegenome
rundb_results
rundb_rig
rundb_tfmetrics
)
ONE_EXPECTED_LIST=(
auth_group_id_seq
auth_group_permissions_id_seq
auth_message_id_seq
auth_permission
auth_permission_id_seq
auth_user
auth_user_groups_id_seq
auth_user_id_seq
auth_user_user_permissions_id_seq
django_admin_log_id_seq
django_content_type
django_content_type_id_seq
django_site
django_site_id_seq
rundb_analysismetrics_id_seq
rundb_backup_id_seq
rundb_backupconfig_id_seq
rundb_chip
rundb_chip_id_seq
rundb_cruncher_id_seq
rundb_emailaddress_id_seq
rundb_experiment_id_seq
rundb_fileserver
rundb_fileserver_id_seq
rundb_globalconfig
rundb_globalconfig_id_seq
rundb_libmetrics_id_seq
rundb_location
rundb_location_id_seq
rundb_plugin_id_seq
rundb_referencegenome_id_seq
rundb_reportstorage
rundb_reportstorage_id_seq
rundb_results_id_seq
rundb_runscript
rundb_runscript_id_seq
rundb_template
rundb_template_id_seq
rundb_tfmetrics_id_seq
)
    
#---	Test for required tool	---#
if ( ! which /usr/bin/psql ); then
	echo "/usr/bin/psql is not installed.  Exiting $0"
	exit 1
fi

if [ -f /opt/ion/.computenode ]; then
    echo "This is not the head node.  Exiting $0"
    exit 1
fi

echo "0 or more expected"
for n in ${ZERO_EXPECTED_LIST[@]}; do
    echo $n
    /usr/bin/psql -U ion -d iondb -c "select count(*) from $n"|grep -A 1 "\-------"|tail -1
done
echo "=================================================================="
echo "1 or more expected"
for n in ${ONE_EXPECTED_LIST[@]}; do
    echo $n
    /usr/bin/psql -U ion -d iondb -c "select count(*) from $n"|grep -A 1 "\-------"|tail -1
done

exit 0
