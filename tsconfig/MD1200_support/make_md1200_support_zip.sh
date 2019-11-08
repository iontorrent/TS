#!/bin/bash
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
set -e
set -u
date=$(date +%Y%m%d)

#==============================================================================
# Download required packages
#==============================================================================
function download_prerequisites()
{
    destdir=$1
    sudo apt-get clean   # Need to clean the cache in order to get the uris
    required=( $(get_required_pkgs) )
    
    for pkgname in ${required[@]}; do

        URL=$(apt-get install --reinstall --assume-yes --allow-unauthenticated --print-uris $pkgname|tail -1|awk -F\' '{print $2}')
        wget --directory-prefix=./${destdir} $URL
    done
}
#==============================================================================
# Return list of required packages based on distribution
#==============================================================================
function get_required_pkgs()
{
    if [ "$(lsb_release -cs)" == "trusty" ]; then
        echo "libdevmapper-event1.02.1 watershed megacli dkms lvm2 xfsprogs lsscsi"
    else
        echo "libdevmapper-event1.02.1 watershed megacli dkms lvm2 xfsprogs lsscsi megaraid-sas-dkms"
    fi
}

# Clear the deck and repopulate from scratch everytime
builddir=md1200_support
rm -rf ${builddir}
mkdir ${builddir}

# Get required deb packages
download_prerequisites ${builddir}

# Get latest TSaddstorage script
cp -vp ../bin/TSaddstorage ${builddir}
cp -vp ./README.txt ${builddir}
cp -vp ./stresstest.tgz ${builddir}
cp -vp ./driver_update ${builddir}

ZIPFILE=md1200_support_$(lsb_release -cs)_$date.zip
zip -r $ZIPFILE ${builddir}/

md5sum $ZIPFILE > ${ZIPFILE%.*}.checksum

exit 0
