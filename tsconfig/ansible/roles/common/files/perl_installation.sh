#!/bin/bash
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
module=$1
echo "We are going to install $module"

if [[ ! -d /usr/local/lib/perl ]]; then
    PERLPOD=""
    INSTALL_ALL=true
else
    PERLPOD=$(find /usr/local/lib/perl -type f -name perllocal.pod)
    INSTALL_ALL=false
fi

# Determine if module is not installed yet and install if its not
# NOTE: This only checks the perllocal.pod file - it does not verify the modules files exist in the filesystem.
# If the files were somehow removed, but the perllocal.pod file not updated, the module would not get installed.
module_name=$(echo ${module%-*} |sed s/-/::/)
if $INSTALL_ALL || ! grep "$module_name" $PERLPOD -A20 | tail -21 | egrep -q '(installed|VERSION)'; then
    cd $(dirname $module)
    instLog=${module%.tar.gz}_install.log
    echo "Installing ${module%.tar.gz}.  See $instLog"
    tar zxf $module
    cd ${module%.tar.gz}
    perl Makefile.PL > $instLog 2>&1
    make >> $instLog 2>&1
    make install >> $instLog 2>&1
fi
