#!/bin/bash
# Copyright (C) 2013-2014 Ion Torrent Systems, Inc. All Rights Reserved
#
# Script to simultaneously query software version on all compute nodes
# Requires ssh-key be installed on compute nodes for user ionadmin
# Requires ionadmin on compute have sudo privilege without password prompt
#     Add this to sudoers: Defaults:ionadmin !authenticate
set -e
set -u
DEBUG=0

#==============================================================================
# Include function definition file
#==============================================================================
TSCONFIG_SRC_DIR='/usr/share/ion-tsconfig'
source $TSCONFIG_SRC_DIR/ts_params
source $TSCONFIG_SRC_DIR/grp_functions
SCRIPT=$(basename $0)
ERROR_LOG=${TSCONFIG_LOG_DIR}/${SCRIPT%.sh}.error
LOG_LOG=${TSCONFIG_LOG_DIR}/${SCRIPT%.sh}.log

#==============================================================================
# MAIN
#==============================================================================
HOST_LIST=( `get_crunchers` )

if [ -u $HOST_LIST ]; then
    echo "foo.  no hostnames in the host_list"
    exit
fi

HOST_ARRAY=( `echo $HOST_LIST|sed 's/,/ /g'` )
echo "${HOST_ARRAY[@]}"

set +e  # disable exit on error for now.

# Override default ssh parameters used by pdsh
export PDSH_SSH_ARGS="-o StrictHostKeyChecking=no -2 -a -x -l%u %h"
CLUST_CMD="$TOOL -l $USER -S -w ssh:$HOST_LIST "
MNT_VER_CMD="${CLUST_CMD}mountpoint /results"
PKG_VER_CMD="${CLUST_CMD}dpkg -l "

#==============================================================================
# Check compute nodes for mounted drive at /results
#==============================================================================
$MNT_VER_CMD  2> ${ERROR_LOG} 1> ${LOG_LOG}
if [[ $? -ne 0 ]]; then
	echo THERE WAS A NON_ZERO SOMEWHERE
fi

#==============================================================================
# Verify the software versions
#==============================================================================
$PKG_VER_CMD "ion-analysis|tail -1" 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
$PKG_VER_CMD "ion-gpu|tail -1" 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
$PKG_VER_CMD "ion-pipeline|tail -1" 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
$PKG_VER_CMD "ion-rsmts|tail -1" 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
$PKG_VER_CMD "ion-torrentr|tail -1" 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
$PKG_VER_CMD "ion-tsconfig|tail -1" 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
$PKG_VER_CMD "ion-docs|tail -1" 2>> ${ERROR_LOG} 1>> ${LOG_LOG}

${CLUST_CMD}sudo apt-get -qq update 2>> ${ERROR_LOG}

#==============================================================================
# Print log, per host
#==============================================================================
for host in ${HOST_ARRAY[@]}; do
	echo "==============="
    echo " $host"
	echo "==============="
	grep ${host}: ${LOG_LOG}||true
	grep ${host}: ${ERROR_LOG}||true
done

exit
