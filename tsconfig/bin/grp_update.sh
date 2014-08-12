#!/bin/bash
# Copyright (C) 2013-2014 Ion Torrent Systems, Inc. All Rights Reserved
#
# Script to simultaneously update software on all compute nodes
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
APT_UPDATE_CMD="${CLUST_CMD}sudo apt-get -qq update"
if [ $DEBUG == 0 ]; then
    # DEPLOY VERSION
    TSCONFIG_INSTALL_CMD="${CLUST_CMD}sudo apt-get install --force-yes -y ion-tsconfig"
    TSCONFIG_CMD="${CLUST_CMD}sudo TSconfig -s"
else
    #DEBUG VERSION
    TSCONFIG_INSTALL_CMD="${CLUST_CMD}sudo echo POOPS TO YOU: don\'t run ion-tsconfig install"
    TSCONFIG_CMD="${CLUST_CMD}sudo echo POOPS TO YOU: don\'t run TSconfig -s"
fi

#==============================================================================
# Update apt repository
#==============================================================================
echo "Running the following command: $APT_UPDATE_CMD"
$APT_UPDATE_CMD  2> ${ERROR_LOG} 1> ${LOG_LOG}
if [[ $? -ne 0 ]]; then
	echo THERE WAS AN ERROR SOMEWHERE
fi

#==============================================================================
# Update ion-tsconfig package
#==============================================================================
echo "Running the following command: $TSCONFIG_INSTALL_CMD"
$TSCONFIG_INSTALL_CMD 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
if [[ $? -ne 0 ]]; then
	echo THERE WAS AN ERROR SOMEWHERE
fi

#==============================================================================
# Update Torrent Suite
#==============================================================================
echo "Running the following command: $TSCONFIG_CMD"
$TSCONFIG_CMD 2>> ${ERROR_LOG} 1>> ${LOG_LOG}
if [[ $? -ne 0 ]]; then
	echo THERE WAS AN ERROR SOMEWHERE
fi

#==============================================================================
# Print log, per host
#==============================================================================
if [ $DEBUG -eq 1 ]; then
    for host in ${HOST_ARRAY[@]}; do
        echo "==============="
        echo " $host"
        echo "==============="
        grep ${host}: ${LOG_LOG}||true
        grep ${host}: ${ERROR_LOG}||true
    done
fi

exit
