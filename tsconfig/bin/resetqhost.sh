#!/bin/bash
# Copyright (C) 2019 Thermo Fisher Scientific, Inc. All Rights Reserved
#
# Purpose: reset SGE hostname
# 
# Usage: resetqhost.sh
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

