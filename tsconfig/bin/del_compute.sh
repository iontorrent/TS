#!/bin/bash
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
# Purpose: Remove compute node from cluster.
# Use Case 1: Take down the compute to reassign to a different cluster
# Use Case 2: Node is bad and must be taken out of service
set -u
set -e

#---                                    ---#
#--- Include function definition file	---#
#---                                    ---#
TSCONFIG_SRC_DIR='/usr/share/ion-tsconfig'
source $TSCONFIG_SRC_DIR/ts_params 2>/dev/null||source ../ts_params||true
source $TSCONFIG_SRC_DIR/ts_functions 2>/dev/null|| source ../ts_functions||true
DEBUG=${DEBUG-False}

# Test for permission
needs_root

# Do something
myhost=$1

# Playbook to remove Ion and gridengine pkgs and unmount shared drives
yellow;echo -e "\nN.B. If the node is unreachable, the deconfigure_compute playbook will show errors.\n";clrclr
cd "$ANSIBLE_HOME"
if ! ansible-playbook -i "$ANSIBLE_HOME"/"$MY_HOSTS" deconfigure_compute.yml --become --limit="$myhost"; then
    red;echo -e "\nNode is unreachable.  Cleanup will have to be manual.\n";clrclr
    echo -e "\t* Uninstall Torrent Suite software"
    echo -e "\t* Uninstall gridengine software"
    echo -e "\t* Purge /var/lib/gridengine/iontorrent/common/act_qmaster"
    echo -e "\t* Unmount mountpoints and remove entries from /etc/fstab"
fi

# Remove host from gridengine configuration
yellow;echo -e "\nCleaning up gridengine entries...\n";clrclr
sge_delete_host "$myhost"

# Remove host from compute list
yellow;echo -e "\nCleaning up $myhost from $ANSIBLE_HOME/$MY_HOSTS\n";clrclr
sed -i /"$myhost"/d "$ANSIBLE_HOME"/"$MY_HOSTS"

