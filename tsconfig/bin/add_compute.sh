#!/bin/bash
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
#
# all arguments are treated as compute nodenames to be added to the cluster
#
#---                                    ---#
#--- Include function definition file	---#
#---                                    ---#
TSCONFIG_SRC_DIR='/usr/share/ion-tsconfig'
source $TSCONFIG_SRC_DIR/ts_params 2>/dev/null||source ../ts_params||true
source $TSCONFIG_SRC_DIR/ts_functions 2>/dev/null|| source ../ts_functions||true
DEBUG=${DEBUG-False}

#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------
function is_valid_compute()
{
    # Node validation.  does it exist, can we talk to it, something else?
    if [ "$1" == "test3" ]; then
        return 1
    fi
    
    if ! ping -c3 -l3 $1 >/dev/null; then
        echo -n "Failed to ping "
        return 1
    fi
    
    return 0
}

#-------------------------------------------------------------------------------
# Check for valid distribution
# Only Ubuntu 14.04 allowed at this time
#-------------------------------------------------------------------------------
if [ $(lsb_release -r -s) == '14.04' ]; then
    #echo "Valid OS"
    :
else
    echo "This OS distribution not supported."
    exit 0
fi

#-----------------------------------------------------------------------
# Check for arguments and print help
#-----------------------------------------------------------------------
if [ $# == 0 ]; then
    echo
    echo "Usage: add_compute.sh [nodename[ nodename]...]"
    echo
    echo "Provide one or more hostnames separated by a space"
    echo
    echo "Each nodename will be added to sge as admin host, added to the [computes] list in torrentsuite_hosts_local,"
    echo "and then ansible-playbook to configure each server will be run."
    echo
    exit
fi

#--------------------------------------
# Must be root
#--------------------------------------
needs_root

#-----------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------
echo "=================================================================="
echo "Compute Node Configuration Commencing..."
echo "=================================================================="
nodelist=
while [ $# != 0 ]; do
    if ! is_valid_compute $1; then
        echo "$1 is not a valid compute node.  Skipping."
        :
    else
        #----------------------------------------------------------
        # Make the node an admin host
        #----------------------------------------------------------
        cmd="qconf -ah $1"
        if [ $DEBUG == True ]; then
            echo "CMD: $cmd"
        else
            $cmd
        fi
        
        #----------------------------------------------------------
        # Update torrentsuite_hosts_local
        #----------------------------------------------------------
        if ! grep -q "^$1$" ${ANSIBLE_HOME}/${MY_HOSTS}; then
            # Insert the node right after the [computes] tag
            cmd="sed -i s/\[computes\]/\[computes\]\n$1/ ${ANSIBLE_HOME}/${MY_HOSTS}"
            if [ $DEBUG == True ]; then
                echo "CMD: $cmd"
            else
                $cmd
            fi
        fi

        #-----------------------------------------------------------------------
        # copy secure key to the compute node
        #-----------------------------------------------------------------------
        echo "=================================================================="
        echo "Copy ssh key to: $1"
        echo "=================================================================="
        if [ $DEBUG == True ]; then
            echo "CMD: ssh-keygen, ansible -m authorized_key"
        else
            # Make sure we have an ssh key to give to the compute nodes - Run as ionadmin user, not root
            sudo -u $TSCONFIG_CLUSTER_ADMIN mkdir --mode=0700 -p /home/$TSCONFIG_CLUSTER_ADMIN/.ssh
            ANSIBLE_RSA_KEY=/home/$TSCONFIG_CLUSTER_ADMIN/.ssh/ansible_rsa_key
            if [[ ! -e ${ANSIBLE_RSA_KEY} ]]; then
                log "Generate ssh key"
                sudo -u $TSCONFIG_CLUSTER_ADMIN ssh-keygen -q -f ${ANSIBLE_RSA_KEY} -t rsa -N ''
            fi
            # This prompts for a password every time.
            echo -e "\nEnter compute node password..."
            ansible $1 -i ${ANSIBLE_HOME}/${MY_HOSTS} -m authorized_key -a "user=$TSCONFIG_CLUSTER_ADMIN key='{{ lookup('file', '${ANSIBLE_RSA_KEY}.pub') }}' path=/home/$TSCONFIG_CLUSTER_ADMIN/.ssh/authorized_keys manage_dir=no" --ask-pass -c paramiko
    
        fi
        
        #----------------------------------------------------------
        # Add to list of computes for ansible playbook
        #----------------------------------------------------------
        if [ "$nodelist" != "" ]; then
            nodelist=(${nodelist[@]},$1)
        else
            nodelist=($1)
        fi
    fi
    shift
done

# Check for valid nodes to process
if [ -z $nodelist ]; then
    echo "Empty list. Exiting"
    exit
fi


#-----------------------------------------------------------------------
# ansible configuration
#-----------------------------------------------------------------------
echo "=================================================================="
echo "Add Compute Node: $nodelist"
echo "=================================================================="
cd $ANSIBLE_HOME
cmd="ansible-playbook site.yml -i ${ANSIBLE_HOME}/${MY_HOSTS} --become --limit=${nodelist[@]}"
if [ $DEBUG == True ]; then
    echo "CMD: $cmd"
else
    $cmd
fi

exit
