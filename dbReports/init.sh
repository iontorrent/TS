#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
echo 'sudo apt-get install python-setuptools'
sudo apt-get install python-setuptools

echo 'sudo easy_install -U pip'
sudo easy_install -U pip

echo 'sudo pip install virtualenv==1.8.2'
sudo pip install virtualenv==1.8.2

echo 'rm -rf local-python'
rm -rf local-python

echo 'virtualenv --system-site-packages local-python'
virtualenv --system-site-packages local-python

echo 'source local-python/bin/activate'
source local-python/bin/activate

echo 'pip install Fabric==1.4.3'
pip install Fabric==1.4.3

echo 'fab dev_setup'
fab dev_setup

echo '!!!!!!!!!!!!!!!!!! Activate the isolated python environment !!!!!!!!!!!!!!!!!!!'
echo 'Execute the following within your terminal window:'
echo 'source local-python/bin/activate'
