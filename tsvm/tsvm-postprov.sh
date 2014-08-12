#!/bin/bash
# Copyright (C) 2013-2014 Ion Torrent Systems, Inc. All Rights Reserved
#
# This script is executed in the VM environment during/after provisioning
#

#=========================================================================
# Functions
#=========================================================================
makeVarsFromJson()
{
    local IFS=$'\n'
    for line in $(jsonpipe < $1);do
        if echo $line|egrep -q '(\{}|\[])'; then
            :
        else
            line=$(echo $line|sed 's:^/::g')
            var_name=$(echo $line|awk '{print $1}')
            var_val=$(echo $line|awk -F'\t' '{print $2}'|sed 's/"//g')
            index=$(basename $var_name)

            # var_name ends in a number, its an array variable
            if [ $index -eq $index 2>/dev/null ]; then
                #strip off number
                var_name=$(dirname $var_name)
                #convert slashes to double underscore
                #convert to uppercase
                var_name=$(echo ${var_name^^}|sed 's:/:__:g')
                #echo $var_name[$index]=$var_val
                eval $var_name[$index]=\""$var_val"\"
                export ${var_name}
            else
                #convert slashes to double underscore
                #convert to uppercase
                var_name=$(echo ${var_name^^}|sed 's:/:__:g')
                #echo $var_name=$var_val
                export eval "${var_name}"="${var_val}"
            fi

        fi 
    done
}

getRigs ()
{
    #NOTE: Only print names of rigs from this function!
    local HOST=$1
    local TMP_OUT=$(mktemp)
    #NOTE: there is a twenty object limit.  This will only get the first 20 objects
    wget --output-document=${TMP_OUT} --output-file=/tmp/wget.errors http://ionadmin:ionadmin@${HOST}/rundb/api/v1/rig/?format=json
    exitcode=$?

    if [ $exitcode != 0 ]; then
        echo "ERROR getting names of rigs from host TS.  See /tmp/wget.errors" >&2
	return
    fi
    
    makeVarsFromJson ${TMP_OUT}

    num_objs=$(printenv|grep META__TOTAL_COUNT|awk -F= '{print $2}')
    if [ -z $num_objs ]; then
    	return
    fi
    iter=0
    while (( $iter < ${num_objs} )); do
    	#printenv|grep "OBJECTS__${iter}__NAME"
    	name=$(printenv|grep "OBJECTS__${iter}__NAME"|awk -F= '{print $2}')
    	pgm_list+=( $name )
    	iter=$((iter+1))
    done
    rm -f ${TMP_OUT}
    
    echo ${pgm_list[@]}
}

transfer_reference_genomes()
{
    local HOST=$1
    echo "Looking for Reference Genomes to transfer..."
python << EOF
import os
import sys
import errno
import requests
from requests.auth import HTTPBasicAuth
from iondb.bin import djangoinit
from iondb.rundb.models import ReferenceGenome
response = requests.get(\
           'http://ionadmin:ionadmin@${HOST}/rundb/api/v1/referencegenome/', \
            auth=HTTPBasicAuth('ionadmin','ionadmin'),\
            params={'format':'json'})
#Test for a valid response
try:
    response.json()
except Exception as e:
    print e
    sys.exit()
for i, refobj in enumerate(response.json()['objects']):
    print "(%d) %s" % (i+1, refobj['short_name'])
    kwargs = {}
    for key, arg in refobj.iteritems():
        if 'resource_uri' in key:   # TS4.2 and newer no longer contain this field
            continue
        if 'pk' in key or 'id' in key:  # Let database assign a unique pk
            continue
        kwargs[key] = arg

    # Copy from /results_host (Native TS) to /results (TS-VM)
    src = kwargs['reference_path'].replace('/results/', '/results_host/')
    dst = kwargs['reference_path']
    kwargs['reference_path'] = src
    try:
        print "Linking to reference files..."
        os.symlink(src, dst)
        print "Reference Copied: %s" %(kwargs['name'])
    except Exception as e:
        if e.errno == errno.EEXIST:
            print e
        else:
            #raise
            print e
            continue
    
    newobj, created = ReferenceGenome.objects.get_or_create(short_name=kwargs['short_name'], defaults=kwargs)
    if created:
        print "Reference object created: %s" %(kwargs['name'])
    else:
        print "Reference object retrieved: %s" %(kwargs['name'])
EOF

}


addThisFileServer ()
{
FSNAME=$1
python << END
from iondb.bin import djangoinit
from iondb.rundb.models import FileServer, Location
loc_obj = Location.objects.get(defaultlocation=True)
loc = loc_obj
obj, created = FileServer.objects.get_or_create(filesPrefix="$FSNAME", name='TS Host $FSNAME', location=loc)
END
if [ $? -eq 0 ]; then
    echo "Added $FSNAME to File Server table"
fi
}

addThisRig ()
{
RIGNAME=$1
python << END
from iondb.bin import djangoinit
from iondb.rundb.models import Rig, Location
loc_obj = Location.objects.get(defaultlocation=True)
loc = loc_obj
obj, created = Rig.objects.get_or_create(name="$RIGNAME", location=loc)
END
if [ $? -eq 0 ]; then
    echo "Added $RIGNAME to Rig table"
fi
}

defaultLocation ()
{
LOCNAME=$1
python << END
from iondb.bin import djangoinit
from iondb.rundb.models import Location
obj = Location.objects.get(defaultlocation=True)
obj.name = "$LOCNAME"
obj.save()
END
if [ $? -eq 0 ]; then
    echo "Added $LOCNAME to Location table"
fi
}

defaultReportStorage ()
{
RSNAME=$1
python << END
from iondb.bin import djangoinit
from iondb.rundb.models import ReportStorage
obj = ReportStorage.objects.get(default=True)
obj.dirPath = "$RSNAME"
obj.save()
END
if [ $? -eq 0 ]; then
    echo "Added $RSNAME to Location table"
fi
}

setTSName ()
{
TSNAME=$1
python << END
from iondb.bin import djangoinit
from iondb.rundb.models import GlobalConfig
GC = GlobalConfig.objects.get()
GC.site_name = "TS at $TSNAME"
GC.save()
END
if [ $? -eq 0 ]; then
    echo "Modified Torrent Server name to $TSNAME"
fi
}

#=========================================================================
# DEBUG Stuff
#=========================================================================
echo "Command line args: $@"
HOST_HOSTNAME=${1-NoName}
TS_VERSION=${2-'4.2'}
TIMEZONE=${3-America/New_York}
BASIC_SERVER=${4-false}
echo "arg1: '$1'. HOSTNAME=$HOST_HOSTNAME"
echo "arg2: '$2'. TS_VERSION=$TS_VERSION"
echo "arg3: '$3'. TIMEZONE=$TIMEZONE"
echo "arg4: '$4'. BASIC=$BASIC_SERVER"

#=========================================================================
# Set TSVM timezone to same as native
#=========================================================================
echo $TIMEZONE > /etc/timezone
(
    cd /etc/
    ln -s -f /usr/share/zoneinfo/`cat timezone` localtime
)

#==========================================================================
# Update VM database with unique name for TS (default is "Torrent Server")
#==========================================================================
setTSName ts-vm-${HOST_HOSTNAME} || true

#==========================================================================
# Basic standalone Torrent Server, end it here
#==========================================================================
if $BASIC_SERVER; then
    exit
fi


#=========================================================================
# Disable ionCrawler and prevent from ever starting
#=========================================================================
service ionCrawler stop
update-rc.d -f ionCrawler remove
update-rc.d ionCrawler stop 97 0 1 2 3 4 5 6 .

#=========================================================================
# Create subdirectory in /results to contain these reports
#=========================================================================
REPORT_DIR=/results_host/analysis/output
# Make sure group has write permission on /results/analysis/output
chmod g+w $REPORT_DIR || true

DEF_FOLDER="TSVM${TS_VERSION}"
defaultLocation "$DEF_FOLDER"
mkdir -p ${REPORT_DIR}/$DEF_FOLDER
chown www-data.www-data ${REPORT_DIR}/$DEF_FOLDER || true
chmod 0775 ${REPORT_DIR}/$DEF_FOLDER || true

#=========================================================================
# Update VM database with File Server objects for /results_host and /rawdata_host directory
#=========================================================================
addThisFileServer /results_host/
addThisFileServer /rawdata_host/

#=========================================================================
# Update VM database with default Report Storage object 
# Note: This updates the default object; does not create new one
#=========================================================================
defaultReportStorage ${REPORT_DIR}
(
    cd /var/www
    rm -f ./output # This is a link
    ln -s ${REPORT_DIR} output
)

#=========================================================================
# Update VM database with Rig objects that are listed in native TS database.
#=========================================================================
#echo "Getting instrument names from the $HOST_HOSTNAME Torrent Server..."
#list=( `getRigs $HOST_HOSTNAME` )
#echo ${list[@]}
#for item in ${list[@]}; do
#    if [ "$item" == "default" ]; then
#        continue
#    fi
#    # Add the rigname to Rigs table
#    addThisRig $item || true
#done
##DO NOT NEED TO ADD RIGS B/C WE USE IMPORT FUNCTION TO ADD DATASETS

#=========================================================================
# Import Experiments into VM database.
#=========================================================================
if [ -e /opt/ion/iondb/bin/import_runs_from_json.py ]; then
    /opt/ion/iondb/bin/import_runs_from_json.py >/dev/null
fi

#==========================================================================
# vboxsf filesystem type needs to be treated as an archive location
# (ONLY NEEDED IF WE DEPLOY WITH VBOXSF FILESYSTEM)
#==========================================================================
VBOXSF=false
if [[ $VBOXSF ]]; then
patch -N /opt/ion/iondb/utils/devices.py << EOF
Index: devices.py
===================================================================
--- devices.py	(revision 82729)
+++ devices.py	(working copy)
@@ -66,7 +66,7 @@
         # Report Data Management requires an ext3/4 filesystem or nfs (anything that supports symbolic links actually)
         #if 'media' in path and ('ext' in type or 'nfs' in type):
         #if 'nfs' in type or ('/media' in path) or ('/mnt' in path):
-        if 'nfs' in type or path.startswith('/media') or path.startswith('/mnt'):
+        if 'nfs' in type or 'vboxsf' in type or path.startswith('/media') or path.startswith('/mnt'):
             try:
                 if os.path.exists(os.path.join(path, '.not_an_archive')):
                     continue
EOF
service apache2 restart
fi

#==========================================================================
# Lock current TS version on the VM
#N.B. This will mess up sources.list for internal machines running pre-released versions.
#Since the archive location will not exist.
#==========================================================================
#/usr/bin/wget -o /tmp/version_lock.log -O /dev/null http://localhost/admin/update/version_lock/enable_lock
#echo "Torrent Suite is locked at version: $TS_VERSION"


#==========================================================================
# Copy Genome References from native TS to TS-VM
#==========================================================================
transfer_reference_genomes $HOST_HOSTNAME


#==========================================================================
# Verify that all required settings are correct
#==========================================================================
# Check /var/www/output links to /results_host
if [ $(readlink /var/www/output) == "${REPORT_DIR}" ]; then
    echo "CORRECT: /var/www/output points to ${REPORT_DIR}"
else
    echo 'ERROR: /var/www/output is not configured properly'
    ls -l /var/www
fi
# /results_host/analysis/output/TSVM exists
TEST_DIR=${REPORT_DIR}/${DEF_FOLDER}
if [ -d ${TEST_DIR} ]; then
    echo "CORRECT: path exists: ${TEST_DIR}"
else
    echo "ERROR: path does not exist: ${TEST_DIR}"
fi
# /results_host/analysis/output/TSVM4.2 writeable by www-data user
if sudo -u www-data touch ${TEST_DIR}/test; then
    echo "CORRECT: www-data has write permission in: ${TEST_DIR}"
    sudo -u www-data rm -f ${TEST_DIR}/test
else
    echo "ERROR: www-data does not have write permission in: ${TEST_DIR}"
fi

exit
