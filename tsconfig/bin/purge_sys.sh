#!/bin/bash
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

#
# TODO: Unmount remote mount directories
# TODO: Backup /etc/fstab remote mount directives
#
# TODO: Backup postgres database
# TODO: Backup iptables.custom
# TODO: Backup /etc/hosts
# TODO: Remove iptables commands from /etc/hosts
#

# ????: Are the daemons shutdown prior to unconfiguring?


# Release package holds
dpkg --get-selections > /tmp/pkgs_installed
sed -i 's/\shold.*/install/' /tmp/pkgs_installed
cat /tmp/pkgs_installed | dpkg --set-selections
rm -f /tmp/pkgs_installed



#
# Purges packages in preparation for a distribution upgrade
# List taken from torrentsuite_packagelist.json and edited.
#
pkgs=(
traceroute
ethtool
arp-scan
nmap
imagemagick
vcftools
bedtools
curl
whois
figlet
tk8.5
libmotif-dev
libxpm-dev
libboost-all-dev
xorg
xfce4
gridengine-common
gridengine-client
gridengine-exec
libdrmaa1.0
iptables
ntp
nfs-kernel-server
samba
libz-dev
libxml2-dev
postfix
python-pysam
python-simplejson
python-calabash
python-jsonpipe
python-rpy2
python-django
python-gnuplot
python-requests
python-httplib2
perl-doc
default-jre
gnuplot-x11
putty-tools
tmpreaper
timelimit
gridengine-master
gridengine-qmon
rabbitmq-server
vsftpd
postgresql
apache2-mpm-prefork
apache2
libapache2-mod-wsgi
libapache2-mod-php5
dnsmasq
dhcp3-server
tomcat6
tomcat6-admin
nxclient
nxnode
nxserver
python-amqp
)

for pkg in ${pkgs[@]}; do
    apt-get remove --purge --assume-yes --force-yes $pkg
done

apt-get autoremove --assume-yes


# Remove Ion repositories
sed -i '/ionupdates.com/d' /etc/apt/sources.list
sed -i '/updates.iontorrent.com/d' /etc/apt/sources.list
sed -i '/updates.ite/d' /etc/apt/sources.list
sed -i '/updates.itw/d' /etc/apt/sources.list
sed -i '/updates.cbd/d' /etc/apt/sources.list

# Edit release-upgrade
sed -i 's/prompt.*/prompt=lts/' /etc/update-manager/release-upgrades

#Technically, can now upgrade
#do-release-upgrade -f DistUpgradeViewNonInteractive
echo -e "y\ny\ny\ny\ny\ny\ny\ny\n" | do-release-upgrade -f DistUpgradeViewNonInteractive
exit
