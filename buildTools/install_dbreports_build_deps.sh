#!/bin/bash
# Copyright (C) 2021 Ion Torrent Systems, Inc. All Rights Reserved

# This will install a build dependency for dbReports, Node.js and Recess

sudo apt-get install cmake build-essential curl openssl libssl-dev
mkdir -p /tmp/nodejs
cd /tmp/nodejs
wget http://nodejs.org/dist/v0.8.0/node-v0.8.0.tar.gz
tar xzf node-v0.8.0.tar.gz
cd /tmp/nodejs/node-v0.8.0/
./configure
make -j2
sudo make install
if [ $? -eq 0 ]; then
    sudo npm install recess uglify-js jshint hogan.js -g
fi
rm -rf /tmp/nodejs
which recess
if [ $? -eq 0 ]; then
    echo "SUCCESS    You can now build dbReports."
else
    echo "FAIL!    Something went wrong during installation."
fi
