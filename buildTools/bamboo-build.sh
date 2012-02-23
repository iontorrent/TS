#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

EC2_PRIVATE_KEY=/root/pk.pem
EC2_CERT=/root/cert.pem

MODULES=${MODULES-"
	Analysis
 	alignTools
	dbReports
	ionifier
	tsconfig
    	gpu
"}
export MODULES

show_help() {
    echo "Usage: $0"
    echo "  -t tag      Build Tag (eg 'DEV', 'TRUNKCI')"
    echo "  -n num      Build Number (eg '342')"
    echo
    echo " Options related to successful build:"
    echo "  -i          Install packages"
    echo "  -a ddclient config file name"
}

initialize_system_users() {
    echo "Initializing required users and groups"

    sudo addgroup --system ionian --gid 1100 
    sudo adduser --system --no-create-home --uid 1100 --ingroup ionian --disabled-password ionian 
    sudo adduser ionadmin ionian
    sudo adduser www-data ionian
    adduser ionadmin www-data 
}


BUILDTAG="dev"
BUILDNUM="0"
SETDNS=
DOINSTALL=0

while getopts ":t:n:ia:" opt; do
    case $opt in
        t ) BUILDTAG=$OPTARG;;
        n ) BUILDNUM=$OPTARG;;
        i ) DOINSTALL=1;;
        a ) SETDNS=$OPTARG;;
        \? ) show_help; exit;;
    esac
done;
shift $(($OPTIND - 1))

# Update version of TMAP installed 
# commenting out tmap pkg install RB 12/10/11
TMAP_VERSION=`tmap --version | grep Version | awk '{print $2}'`
#if [ "$TMAP_VERSION" != "0.0.25" ]; then
#        wget  http://mendel.iontorrent.com/vm/tmap_0.0.25-1_amd64.deb
#        sudo dpkg -i tmap_0.0.25-1_amd64.deb
#        rm tmap_0.0.25-1_amd64.deb
#fi

# commenting out samtools pkg install   RB 11/7/11
SAMTOOLS_VERSION=$(samtools 2>&1| grep ^Version: | awk '{print $2}')
#if [ "$SAMTOOLS_VERSION" != "0.1.16" ]; then
#	wget http://mendel.iontorrent.com/vm/samtools_0.1.16-1~lucid1_amd64.deb
#    sudo dpkg -i samtools_0.1.16-1~lucid1_amd64.deb
#    rm samtools_0.1.16-1~lucid1_amd64.deb
#fi


# if user ionian does not exist, this is a new build environment 
USERID=ionian
/bin/egrep  -i "^${USERID}" /etc/passwd
if [ $? -eq 0 ]; then
   echo "User $USERID exists in /etc/passwd"
else
   echo "User $USERID does not exist in /etc/passwd"
   initialize_system_users
   echo "installing packages..."
   echo "deb ssh://bamboo@ie.iontorrent.com/home/bamboo/lucid-alpha lucid-alpha/" | sudo tee -a /etc/apt/sources.list.d/ion.list
   sudo apt-get update
   sudo apt-get --assume-yes --force-yes install zeroinstall-injector
   sudo apt-get --assume-yes --force-yes install tmap 
   sudo apt-get --assume-yes --force-yes install ion-pipeline 
fi




buildTools/build.sh -DBUILDTAG=$BUILDTAG -DBUILDNUM=$BUILDNUM

if [ $? != 0 ]; then
    exit 1;
fi

# Make a folder with results (artifacts) that we can easily grab
rm -rf results
mkdir results
cp build-number.txt results/
cp `find build/ -name \*.deb` results


ERR=0

if [ "$DOINSTALL" != 0 ]; then

    echo "=================================================="
    echo " Installing Modules"
    for MODULE in $MODULES; do
        echo -n " $MODULE..."
        sudo dpkg -i build/$MODULE/ion-*.deb 1>install.$MODULE.out 2>&1
        if [ "$?" != 0 ]; then
            echo "Failed."
            cat install.$MODULE.out
            ERR=$(($ERR + 1))
        else
            echo "Done."
        fi
    done;
    echo "=================================================="
fi

# Not working because when we associate a new IP address, we loose the
# original public IP address which then results in bamboo client 
# failing to call home.
if [ "$SETDNS" ]; then
    sudo ddclient -file buildTools/$SETDNS
fi

if [ $ERR != 0 ]; then
    echo "FAILURES: $ERR modules failed to install."
    exit $ERR
fi
