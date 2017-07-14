#!/bin/bash
# Copyright 2016 Thermo Fisher Scientific. All Rights Reserved.

# http://mirror-host/ubuntu -> archive.ubuntu.com
ln -s /var/www/ubuntu /var/spool/apt-mirror/mirror/archive.ubuntu.com/ubuntu/

# http://mirror-host/docker -> apt.dockerproject.org/repo/
ln -s /var/www/docker /var/spool/apt-mirror/mirror/apt.dockerproject.org/repo/

# http://mirror-host/ansible -> ppa.launchpad.net/ansible/ansible/ubuntu/
ln -s /var/www/ansible /var/spool/apt-mirror/mirror/ppa.launchpad.net/ansible/ansible/ubuntu/

