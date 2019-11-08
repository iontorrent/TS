#!/bin/bash
# grab necessary debian files from debian.org since the ones on Ubuntu are broken.

packages=(
gridengine-master_8.1.9+dfsg-8_amd64.deb
gridengine-client_8.1.9+dfsg-8_amd64.deb
gridengine-common_8.1.9+dfsg-8_all.deb
gridengine-exec_8.1.9+dfsg-8_amd64.deb
gridengine-qmon_8.1.9+dfsg-8_amd64.deb
)

optionls=(
gridengine-drmaa1.0_8.1.9+dfsg-8_amd64.deb
gridengine-drmaa-dev_8.1.9+dfsg-8_amd64.deb
)

srcUrl='http://ftp.debian.org/debian/pool/main/g/gridengine/'

for pkg in ${packages[@]}; do
    url="$srcUrl/$pkg"
    wget $url
done
