#!/bin/bash
# Copyright (C) 2021 Ion Torrent Systems, Inc. All Rights Reserved

# This will install a build dependency for dbReports, Node.js and Recess

sudo apt-get install cmake build-essential curl openssl libssl-dev
mkdir -p /tmp/nodejs
cd /tmp/nodejs
wget http://nodejs.org/dist/v0.10.20/node-v0.10.20.tar.gz
tar xzf node-v0.10.20.tar.gz
cd /tmp/nodejs/node-v0.10.20/
./configure
make -j2
sudo make install
if [ $? -eq 0 ]; then
    sudo npm install recess jshint hogan.js -g
    sudo npm install uglify-js@1 -g
fi
rm -rf /tmp/nodejs
which recess
if [ $? -eq 0 ]; then
    echo "SUCCESS    You can now build dbReports."
else
    echo "FAIL!    Something went wrong during installation."
fi
