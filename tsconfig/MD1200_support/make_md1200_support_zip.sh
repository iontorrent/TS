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
    required=( libdevmapper-event1.02.1 watershed megacli dkms megaraid-sas-dkms lvm2 xfsprogs lsscsi )

    for pkgname in ${required[@]}; do

        URL=$(apt-get install --reinstall --assume-yes --force-yes --print-uris $pkgname|tail -1|awk -F"'" '{print $2}')
        wget --directory-prefix=./${destdir} $URL
    done
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

zip -r md1200_support_$date.zip ${builddir}/

md5sum md1200_support_$date.zip > md1200_support_$date.checksum

exit 0
