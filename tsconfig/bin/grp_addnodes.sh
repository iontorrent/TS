#!/bin/bash
# Copyright (C) 2013,2014  Ion Torrent Systems, Inc. All Rights Reserved
#
# Add nodes to the database
# Node names are pulled from SGE host list
# This tool is intended for clusters that have already been created
#

#==============================================================================
# Verify apt-cacher-ng is enabled/installed on headnode
#==============================================================================
if [[ $(dpkg -l apt-cacher-ng 2>/dev/null|tail -1|awk '{print $1}') != 'ii' ]]; then
    echo
    echo "WARNING.  package apt-cacher-ng does not seem to be installed on the headnode."
    echo "Without apt-cacher-ng on the neadnode, the compute nodes will not be able to update deb packages."
    echo
    read -p "Continue with this scrip?(Y|n) " answer
    if [[ "${answer,,}" == "y" ]] || [[ "${answer,,}" == "" ]]; then
        echo "Make sure to do: sudo apt-get install apt-cacher-ng"
        echo "...continuing with the configuration"
        echo
    else
        exit
    fi
fi

#==============================================================================
# Verify this user has an ssh-key file
#==============================================================================
if [[ ! -e ~/.ssh/id_rsa ]]; then
    echo
    echo "$USER does not have an ssh-key generated yet."
    echo "A key is required in order to continue."
    read -p "Do you want to generate one now? (Y|n) " answer
    if [[ "${answer,,}" == "y" ]] || [[ "${answer,,}" == "" ]]; then
        echo
        echo "Hit the enter key at all prompts and do not specify a passphrase:"
        ssh-keygen
        echo
    else
        echo "exiting"
        echo
        exit
    fi
fi
#==============================================================================
# Include function definition file
#==============================================================================
TSCONFIG_SRC_DIR='/usr/share/ion-tsconfig'
source $TSCONFIG_SRC_DIR/grp_functions
SCRIPT=$(basename $0)
HEADNODE=$(hostname)
for node in `get_sge_nodes`; do
    if [[ $node == $HEADNODE ]]; then echo "$node is the headnode"; continue; fi
    node=${node%.*}
    echo "$node"
    read -p "Add this node? [Y|n] " answer
    case $answer in
        'Y'|'y'|'')
            # Adds record to the Crunchers database table
            add_node $node            
            # Adds secure key to node
            ssh-copy-id -i ~/.ssh/id_rsa $USER@$node
            # Adds passwordless commands capability
            scp $TSCONFIG_SRC_DIR/tools/prep_node_sudo.sh $USER@$node:/tmp
            ssh -t $USER@$node "/tmp/prep_node_sudo.sh && rm /tmp/prep_node_sudo.sh"
            # Adds apt cache proxy location
            ssh -t $USER@$node "echo 'Acquire::http::Proxy \"http://$HEADNODE:3142\";'|sudo tee /etc/apt/apt.conf.d/01proxy"
        ;;
        *)
        continue
        ;;
    esac
done

exit
