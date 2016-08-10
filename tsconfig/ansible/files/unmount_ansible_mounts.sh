#!/bin/bash
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
#
# N.B. These variables are also defined in nfs_client.yml and must match.
startline="#start TSconfig added mountpoints"
endline="#end TSconfig added mountpoints"
#sed -n "/^$startline/,/$endline$/p" /etc/fstab

IFS=$'\n'
for entry in $(sed -n "/^$startline/,/$endline$/p" /etc/fstab); do
    if [ "${entry:0:1}" != "#" ]; then
        #echo $entry | awk '{print $2}'
        mntpt=$(echo $entry | awk '{print $2}')
        umount -f -l $mntpt
        rmdir $mntpt
    fi
done

exit 0
