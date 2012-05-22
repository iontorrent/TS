#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

REGEX="^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]))*([A-Za-z]|[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9])$"

IsValidHostname () {
   name=$1
   if [[ $name =~ $REGEX ]] ; then
       return 0
   else
       # Not valid
       return 1
   fi
}


TestThisHostName () {
    testname=$1
    IsValidHostname $testname
    if [ $? == 0 ]; then
        echo "'$testname' is a valid SGE compatible hostname."
    else
        echo "'$testname' is NOT a valid SGE compatible hostname."
    fi
}

echo "Using: " $REGEX
TestThisHostName "hostname"
TestThisHostName "HOSTNAME"
TestThisHostName "HostName"
TestThisHostName "host-name"
TestThisHostName "HOST-NAME"
TestThisHostName "Host-Name"
TestThisHostName "host_name"
TestThisHostName "HOST_NAME"
TestThisHostName "Host_Name"
TestThisHostName "host1name"
TestThisHostName "1hostname"
TestThisHostName "1_hostname"
TestThisHostName "1-hostname"
TestThisHostName "hostname1"
TestThisHostName "hostname-1"
TestThisHostName "hostname_1"
TestThisHostName "hostname.com"
TestThisHostName "host-name.com"
TestThisHostName "host_name.com"
TestThisHostName "HOSTNAME.com"
TestThisHostName "Hostname.com"
TestThisHostName "hostname1.com"
TestThisHostName "hostname_.com"
TestThisHostName "hostname-.com"
TestThisHostName "1hostname.com"
TestThisHostName "_hostname.com"
TestThisHostName "-hostname.com"
TestThisHostName "-hostname"
TestThisHostName "_hostname"
TestThisHostName "hostname-"
TestThisHostName "hostname_"


