#!/bin/bash
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
#set -x 
#set -u
set -e

# Manually increment this after every commit.  Cause I don't know a better way.
VERSION=4.5.12

#--------------------------------------------------------
# List is in start-up priority order.  Shutdown needs to
# traverse in reverse order.
#--------------------------------------------------------
essential_services=(
apache2
celeryd
celerybeat
ionCrawler
ionPlugin
ionJobServer
)
#--------------------------------------------------------
# List of configuration files to record
#--------------------------------------------------------
essential_files=(
/etc/fstab
/etc/network/interfaces
/etc/hostname
/etc/iptables.custom
/etc/iptables.rules
/etc/environment
/etc/apt/sources.list
/etc/apt/apt.conf
/etc/wgetrc
/etc/dhcp3/dhclient.conf
/etc/dhcp3/dhcpd.conf
)

#------------------------------------------------------------
#------------------------------------------------------------
function disable_essential_services()
{
    print_func_head "disable_essential_services"
    for item in ${essential_services[@]}; do
        if [ $DEBUG == False ]; then
            update-rc.d $item disable
        else
            echo "CMD: update-rc.d $item disable"
        fi
    done
}

#------------------------------------------------------------
#------------------------------------------------------------
function enable_essential_services()
{
    print_func_head "enable_essential_services"
    if [ $OK_ENABLE_SERVICES = False ]; then
        echo "ERROR: NOT OKAY TO ENABLE SERVICES AT THIS TIME"
        echo "SCHEMA MIGRATION HAS NOT BEEN RUN"
    else
        echo "These services will start automatically at boot"
        for item in ${essential_services[@]}; do
            if [ $DEBUG == False ]; then
                update-rc.d $item enable
            else
                echo "CMD: update-rc.d $item enable"
            fi
        done
    fi
}

#------------------------------------------------------------
#------------------------------------------------------------
function start_essential_services()
{
    print_func_head "start_essential_services"
    for item in ${essential_services[@]}; do
        echo "Start ${essential_services[idx]}"
        if [ $DEBUG == False ]; then
            service $item restart
        else
            echo "CMD: service $item restart"
        fi
    done
}

#------------------------------------------------------------
#------------------------------------------------------------
function stop_essential_services()
{
    print_func_head "stop_essential_services"
    for (( idx=${#essential_services[@]}-1 ; idx>=0 ; idx-- )) ; do
        echo "Stop ${essential_services[idx]}"
        if [ $DEBUG == False ]; then
            service ${essential_services[idx]} stop
        else
            echo "CMD: service ${essential_services[idx]} stop"
        fi
    done
}

#------------------------------------------------------------
#------------------------------------------------------------
function backup_postgres()
{
    print_func_head "backup_postgres"
    
    echo "Create postgres back up file in /results directory"
    if [ $DEBUG == False ]; then
        pg_dump -U ion -c iondb > /results/postgresql_osupdate.backup
    else
        echo "CMD: pg_dump -U ion -c iondb > /results/postgresql_osupdate.backup"
    fi
}

#------------------------------------------------------------
#------------------------------------------------------------
function restore_postgres()
{
    print_func_head "restore_postgres"
    if [ $OK_RESTORE_DB == False ]; then
        echo "ERROR: NOT OKAY TO RESTORE DATABASE AT THIS TIME"
        echo "DUE TO INCOMPATIBLE DATABASE VERSIONS"
        OK_MIGRATE_SCHEMA=False
    else

        echo "Dropping existing database, and reinitializing iondb database"
        if [ $DEBUG == False ]; then
            # Drop existing database
            echo "Answer y at the prompt"
            sleep 1 # Needed so the stdout lines up properly with the dropdb input request
            dropdb -i -e iondb -U postgres
            sudo -u postgres psql 1>&2 << EOFdb
CREATE DATABASE iondb;
GRANT ALL PRIVILEGES ON DATABASE iondb to ion;
\q
EOFdb
        else
            echo "CMD: dropdb -i -e iondb -U postgres"
            echo "CMD: psql...etc."
        fi
    
        echo "Restoring postgres back up file in /results directory"
        if [ $DEBUG == False ]; then
            psql iondb -U ion < /results/postgresql_osupdate.backup
        else
            echo "CMD: psql iondb -U ion < /results/postgresql_osupdate.backup"
        fi
        OK_MIGRATE_SCHEMA=True
    fi
}

#------------------------------------------------------------
# This function ignores DEBUG flag because it is only backing up files
#------------------------------------------------------------
function backup_essential_files()
{
    _DEBUG=$DEBUG && DEBUG=False
    print_func_head "backup_essential_files"
    if [ $DEBUG == False ]; then
        mkdir -p $STORAGE_DIR
    else
        echo "CMD: mkdir -p $STORAGE_DIR"
    fi
    for item in ${essential_files[@]}; do
        if [ -r $item ]; then
            if [ $DEBUG == False ]; then
                cp -pv --parent $item $STORAGE_DIR
            else
                echo "CMD: cp -pv --parent $item $STORAGE_DIR"
            fi
        else
            echo "Cannot read $item"
        fi
    done
    DEBUG=$_DEBUG
}
#------------------------------------------------------------
# Not all essential files should be copied into place!
#------------------------------------------------------------
function restore_essential_files()
{
    print_func_head "restore_essential_files"
    
    # files that can be copied
    restore_list=(
    /etc/iptables.custom
    /etc/environment
    /etc/apt/apt.conf
    /etc/wgetrc
    )
    for item in ${restore_list[@]}; do
        if [ -e $STORAGE_DIR/$item ]; then
            if [ $DEBUG == False ]; then
                echo "Contents restored: $STORAGE_DIR/$$item --> $item"
                cp -pv $STORAGE_DIR/$item $item
            else
                echo "Contents restored: $STORAGE_DIR/$item --> $item"
                echo "CMD: cp -pv $STORAGE_DIR/$item $item"
            fi
        else
            echo "Not found: $STORAGE_DIR/$item"
        fi
    done
    
    for item in ${essential_files[@]}; do
        if [ -e $item ]; then
            echo "Backup available: $STORAGE_DIR/$item"
        else
            echo "Not found: $STORAGE_DIR/$item"
        fi
    done
}

#------------------------------------------------------------
# Restores all custom mountpoints found in /etc/fstab
#------------------------------------------------------------
function restore_fstab()
{
    print_func_head "restore_fstab"
    oldIFS=$IFS
    IFS=$'\n'
    for line in $(cat $STORAGE_DIR/custom_fstab_entries); do
        # If line is comment, skip it
        if echo $line|grep -q "^#"; then
            continue
        fi
        if [ $DEBUG == False ]; then
            if ! grep -q "$line" /etc/fstab; then
                echo $line >> /etc/fstab
                echo -e "/etc/fstab has been modified...\n$line\nCheck the file before rebooting.\n"
            else
                echo -e "/etc/fstab already contains the line:\n$line"
            fi
        else
            echo "CMD: echo $line >> /etc/fstab"
        fi
        
        mountpoint=$(echo $line|awk '{print $2}')
        
        # Create mountpoint
        if [ ! -e $mountpoint ]; then
            read -p"$mountpoint does not exist. Do you want to create it now? (Y|n) " response
            # Default is yes, if user just hits enter
            if [ "${response,,}" == "y" ] || [ -z $response ]; then
                if [ $DEBUG == False ]; then
                    mkdir $mountpoint
                else
                    echo "CMD: mkdir $mountpoint"
                fi
                mountexists=True
            else
                echo "Skipping creation of $mountpoint"
                mountexists=False
            fi
        else
            echo "$mountpoint exists"
            mountexists=True
        fi
        
        # Mount the entry
        if [ $mountexists == True ]; then
            read -p"Do you want to mount $mountpoint now? (Y|n) " response
            # Default is yes, if user just hits enter
            if [ "${response,,}" == "y" ] || [ -z $response ]; then
                if [ $DEBUG == False ]; then
                    mount $(echo $line|awk '{print $2}') || true
                else
                    echo "CMD: mount $(echo $line|awk '{print $2}')"
                fi
            else
                echo "Skipping mounting $mountpoint"
            fi
        fi

        echo ""
    done
    IFS=$oldIFS

}

#------------------------------------------------------------
#------------------------------------------------------------
function run_migrations()
{
    print_func_head "run_migrations"
    if [ $OK_MIGRATE_SCHEMA = True ]; then
        if [ $DEBUG == False ]; then
            python /opt/ion/manage.py migrate --all
        else
            echo "CMD: python /opt/ion/manage.py migrate --all"
        fi
        OK_ENABLE_SERVICES=True
    else
        echo "ERROR: NOT OKAY TO RUN SCHEMA MIGRATION AT THIS TIME"
        echo "DUE TO THE DATABASE NOT BEING RESTORED YET"
        OK_ENABLE_SERVICES=False
    fi
}

#------------------------------------------------------------
#------------------------------------------------------------
function reset_servername()
{
    print_func_head "reset_servername\nNOTE: Patience, this might take a minute or two"
    if [ "$(hostname)" == "$(cat $STORAGE_DIR/etc/hostname)" ]; then
        echo "Server name was and is '$(hostname)'"
    else
        echo
        echo "The old server name was '$(cat $STORAGE_DIR/etc/hostname)'"
        echo
        if [ $DEBUG == False ]; then
            sudo TSconfig -r
        else
            echo "CMD: TSconfig -r"
        fi
    fi
}

#------------------------------------------------------------
#------------------------------------------------------------
function restore_network()
{
    print_func_head "Restore network settings"
    echo "Editing"
    echo -e "\t/etc/network/interfaces"
    echo -e "\t/etc/dhcp/dhcpd.conf"
    echo -e "\t/etc/dhcp/dhclient.conf"
    echo -e "\t/etc/dnsmasq.d/ion-dhcp"
    # First assumption for T620/T630 systems is that we will only configure
    # the ethernet devices on the 4-port pci add-in card and leave the 2 embedded
    # ports dhcp or unconfigured.
    # 10.04 systems will have eth0 configured as the LAN port.
    # Replace 'eth0' with $default_port (which is typically p4p1 or p5p1 or p6p1)
    # The default LAN port is the first of the 4-port ports
    INTERFACES=$(find $STORAGE_DIR -name interfaces)
    DHCPD=$(find $STORAGE_DIR -name dhcpd.conf)
    DHCLIENT=$(find $STORAGE_DIR -name dhclient.conf)
    IONDHCP=$(find $STORAGE_DIR -name ion-dhcp)
    cp $INTERFACES /tmp/interfaces
    cp $DHCPD /tmp/dhcpd.conf
    cp $DHCLIENT /tmp/dhclient.conf
    cp $IONDHCP /tmp/ion-dhcp
    
    # Map old device names to new on the 4-port pci card only
    for portnum in 0 1 2 3; do
        # Map eth? devices to p?p? device names
        oldname='eth'$portnum
        for filename in /tmp/interfaces /tmp/dhcpd.conf /tmp/dhclient.conf /tmp/ion-dhcp; do
            if grep -q $oldname $filename; then
                newnum=$((portnum+1))
                newname=$(ls /sys/class/net| grep p[0-9]p$newnum) || true
                if [ ! -z $newname ]; then
                    echo "Mapping $oldname ==> $newname"
                    sed -i "s/$oldname/$newname/g" $filename
                else
                    echo "Error!  Where is device em$newnum??"
                    echo "Leaving $oldname device in $(basename $filename)"
                fi
            else
                echo $oldname not found in $(basename $filename) file
            fi
        done
    done
    
    # eth4 and eth5 would be the embedded ports
    for portnum in 4 5; do
        # Map eth? devices to em? device names
        oldname='eth'$portnum
        for filename in /tmp/interfaces /tmp/dhcpd.conf /tmp/dhclient.conf /tmp/ion-dhcp; do
            if grep -q $oldname $filename; then
                newnum=$((portnum-3))
                newname=$(ls /sys/class/net| grep em$newnum) || true
                if [ ! -z $newname ]; then
                    echo "Mapping $oldname ==> $newname"
                    sed -i "s/$oldname/$newname/g" $filename
                else
                    echo "Error!  Where is device em$newnum??"
                    echo "Leaving $oldname device in $(basename $filename)"
                fi
            else
                echo $oldname not found in $(basename $filename) file
            fi
        done
    done
    
    # Install new files
    if [ $DEBUG == False ]; then
        cp -v /tmp/interfaces /etc/network/
        cp -v /tmp/dhcpd.conf /etc/dhcp/
        cp -v /tmp/dhclient.conf /etc/dhcp/
        cp -v /tmp/ion-dhcp /etc/dnsmasq.d/
    else
        echo "CMD: cp -v /tmp/interfaces /etc/network/"
        echo "CMD: cp -v /tmp/dhcpd.conf /etc/dhcp/"
        echo "CMD: cp -v /tmp/dhclient.conf /etc/dhcp/"
        echo "CMD: cp -v /tmp/ion-dhcp /etc/dnsmasq.d/"
    fi
        
    # Restart networking
    lanport=$(ls /sys/class/net| grep p[0-9]p1) || true
    if [ ! -z $lanport ]; then
        if [ $DEBUG == False ]; then
            /sbin/ifdown $lanport && /sbin/ifup $lanport
        else
            echo "CMD: /sbin/ifdown $lanport && /sbin/ifup $lanport"
        fi
    else
        echo "Could not detect default LAN port.  Networking has not been restarted."
    fi
    
    # Restart dhcpd
    if [ $DEBUG == False ]; then
        service isc-dhcp-server restart
    else
        echo "CMD: service isc-dhcp-server restart"
    fi
}

#------------------------------------------------------------
#------------------------------------------------------------
function backup_fstab()
{
    print_func_head "Backup custom mount options"
    oldIFS=$IFS
    IFS=$'\n'
    for item in $(mount | egrep -v '( on /dev| on /run| on /sys| on /proc| on / | on /boot| on /var| on /home| on /lib|/dev/mapper|CLONEZILLA|IMAGE)'); do
        echo $item
        srchterm=$(echo $item|awk '{print $1}')
        custom_entry=$(grep "$srchterm" /etc/fstab) || true
        if [ ! -z "$custom_entry" ]; then
            echo "$custom_entry" >> $STORAGE_DIR/custom_fstab_entries
        else
            echo "$srchterm not found in /etc/fstab"
        fi
    done
    IFS=$oldIFS
    
    TSCONFIG_CARD_ID="PERC H810"
    if grep -q "$TSCONFIG_CARD_ID" /proc/scsi/scsi; then
        print_func_head "MD1200 is installed"
    else
        print_func_head "No MD1200 detected"
    fi
    
    print_func_head "Record installed packages"
    dpkg --get-selections > $STORAGE_DIR/installed_10.04_packages || true
    echo "See $STORAGE_DIR/installed_10.04_packages"
    
    print_func_head "Record crontab entries for ionadmin user"
    crontab -u ionadmin -l > $STORAGE_DIR/crontab_ionadmin && echo "See $STORAGE_DIR/crontab_ionadmin" || true
}

function time_to_shutdown()
{
    print_func_head "OKAY TO REBOOT THE SERVER\nType: sudo reboot"
}

#------------------------------------------------------------
# Delete files in the backup directory.  in case this USB
# is used in multiple locations, we don't want stale files
# This is only executed in the --pre mode!!!
#------------------------------------------------------------
function prep_backup_location()
{
    #print_func_head "prep_backup_location"
    rm -rf $STORAGE_DIR/*
}

#------------------------------------------------------------
# Check and set sun gridengine queue slots
#------------------------------------------------------------
function gridengine_check()
{
    print_func_head "gridengine_check"
    if $(is_t620_lite); then
        if [ $DEBUG == False ]; then
            qconf –aattr queue slots 2 all.q
            qconf –aattr queue slots 2 plugin.q
            qconf –aattr queue slots 2 thumbnail.q
            qconf –aattr queue slots 10 tl.q
        else
            echo "CMD: qconf –aattr queue slots 2 all.q"
            echo "CMD: qconf –aattr queue slots 2 plugin.q"
            echo "CMD: qconf –aattr queue slots 2 thumbnail.q"
            echo "CMD: qconf –aattr queue slots 10 tl.q"
        fi
        echo "queue slots are set for T620 PGM"
    else
        echo "queue slots are set for T620"
    fi
    qstat -f || true
}

#------------------------------------------------------------
# from http://stackoverflow.com/questions/4023830/bash-how-compare-two-strings-in-version-format
#------------------------------------------------------------
function verlte() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

#------------------------------------------------------------
# The ion-dbreports version on old system must be less than
# or equal to version on new system.  If old ion-dbreports is
# newer than the new system, the new system must be software
# updated to greater than or equal to old system version.
#------------------------------------------------------------
function dbreports_version_check()
{
    print_func_head "dbreports_version_check"
    old_sys_version=$(grep "PKG: ion-dbreports" $STORAGE_DIR/pre-migration_report.txt | awk '{print $3}') || true
    if [ -z $old_sys_version ]; then
        echo "I do not know the old version!  It probably wasn't installed."
        OK_RESTORE_DB=False
    else
        echo "Old system's ion-dbreports version was $old_sys_version"
        new_sys_version=$(dpkg -l ion-dbreports | awk "/ion-dbreports/"'{print $3}') || true
        new_sys_version='4.60.1'
        if [ -z $new_sys_version ]; then
            echo "I do not know the new version!  It probably isn't installed."
            OK_RESTORE_DB=False
        else
            echo "New system's ion-dbreports version is $new_sys_version"
            if verlte $old_sys_version $new_sys_version; then
                echo "Okay to proceed with database restore and schema migration"
                OK_RESTORE_DB=True
            else
                echo "TS software needs to be updated before database restore and schema migration"
                OK_RESTORE_DB=False
            fi
        fi
    fi
}

#------------------------------------------------------------
# Import sources.list customizations
# Find the Ion repo URL and check if version is locked
# We check every .list file to make sure we find the right one
# but potentially someone has multiple .list files with Ion
# defined.  All should be the same if they use the GUI to set
# the lock.
#------------------------------------------------------------
function ion_apt_url_check()
{
    print_func_head "ion_apt_url_check"
    TS_VERSION_LOCKED=False
    for listfile in $(find $STORAGE_DIR/etc/apt -name \*.list); do
        echo $listfile
        if grep -q "^deb http://ionupdates.com" $listfile ||\
            grep -q "^deb http://updates" $listfile; then
            if grep -q '^[^#].*updates/software/archive' $listfile; then
                echo "Ion Torrent Suite version is locked"
                TS_VERSION_LOCKED=True
            else
                echo "Ion Torrent Suite version is not locked"
                TS_VERSION_LOCKED=False
            fi
        else
            echo "No Ion apt repo defined"
        fi
    done
    
    if [ $TS_VERSION_LOCKED == True ]; then
        if [ $DEBUG == False ]; then
            TS_VERSION=$(ion_versionCheck.py |awk -F= '/Torrent_Suite/{print $2}')
            echo "deb http://ionupdates.com/updates/software/archive/$TS_VERSION trusty/" > /etc/apt/sources.list.d/iontorrent.list
        else
            TS_VERSION=$(ion_versionCheck.py |awk -F= '/Torrent_Suite/{print $2}')
            echo "CMD: echo \"deb http://ionupdates.com/updates/software/archive/$TS_VERSION trusty/\" > /etc/apt/sources.list.d/iontorrent.list"
        fi
    fi
    
}
#------------------------------------------------------------
#------------------------------------------------------------
function reinstall_md1200()
{
    if grep -q "MD1200 is installed" $STORAGE_DIR/pre-migration_report.txt; then
        print_func_head "!!!! Run TSaddstorage script to restore connectivity with external storage device !!!!"
    else
        :
    fi
}
#------------------------------------------------------------
#------------------------------------------------------------
function set_timezone()
{
    print_func_head "set_timezone"
    echo "Previous timezone: $(cat /etc/timezone)"
    if [ $DEBUG == False ]; then
        dpkg-reconfigure tzdata
    else
        echo "CMD: dpkg-reconfigure tzdata"
    fi
}
#------------------------------------------------------------
#------------------------------------------------------------
function pkg_version()
{
    pkgname=$1
    pkgversion=$(dpkg -l $pkgname | awk "/$pkgname/"'{print $3}')
    print_func_head "PKG: $pkgname $pkgversion"
}

#------------------------------------------------------------
#------------------------------------------------------------
function is_t620_lite()
{
    [ $(dmidecode -s system-product-name|grep -q T620) ] & [ ! -e /rawdata ]
}
#------------------------------------------------------------
#------------------------------------------------------------
function rename_log_directory()
{
    print_func_head "rename_log_directory"
    if [ $DEBUG == False ]; then
        cp -rp $STORAGE_DIR $STORAGE_DIR_$HOSTNAME
    else
        echo "CMD: cp -rp $STORAGE_DIR ${STORAGE_DIR}_$HOSTNAME"
    fi
}
#------------------------------------------------------------
#------------------------------------------------------------
function show_help()
{
    echo
    echo "Usage: osmigration.sh [--pre|--post] [--debug|-d] [--version|-v]"
    echo
    echo -e "\t--pre           Runs the pre-migration steps"
    echo -e "\t--post          Runs the post-migration steps"
    echo -e "\t--dry-run, -d   Dry run.  Does not make any changes to system"
    echo -e "\t--version, -v   Print script version and exit"
    echo
    cat << EOF
    A directory named ${STORAGE_DIR} is created to store all log files and backup
    files.  Run this from the USB key so that it is stored locally.  Otherwise, the
    saved files will be lost during re-imaging.  A copy of the directory is created
    after --post script completes successfully named ${STORAGE_DIR}_${HOSTNAME}.
EOF
    echo
}

#------------------------------------------------------------
# Note: This function can exit the script
#------------------------------------------------------------
function continue_query()
{
    read -p "Continue? [N|y] " myanswer
    if [ "${myanswer,,}" == "y" ]; then
        echo "Proceeding..."
    else
        echo "Exiting..."
        exit
    fi
}

function print_func_head()
{
    echo
    echo "=========================================================="
    echo -e "$1"
    echo "=========================================================="
}

function report_header()
{
    echo "=========================================================="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "=========================================================="
}

function needs_root()
{
    if [ $(id -u) != 0 ]; then
        echo "Please run this script with root permissions:"
        echo
        echo "sudo $0"
        echo
        exit 1
    fi
    return 0
}

function print_success()
{
    echo
    echo "=========================================================="
    echo osmigration.sh completed successfully
    echo "=========================================================="
    echo "See $LOGFILE for a copy of this output."
    echo
}

#------------------------------------------------------------
# Global variables
#------------------------------------------------------------
DEBUG=False
STORAGE_DIR=./os-migration
PRE_MIGRATE=False
POST_MIGRATE=False

#------------------------------------------------------------
#---    First, if no args given, print help ---#
#------------------------------------------------------------
if [ $# == 0 ]; then
    show_help
    exit
fi
#------------------------------------------------------------
#---    We convert all arguments to lower case  ---#
#------------------------------------------------------------
while [ $# != 0 ]; do
    case ${1,,} in
        '--pre')
        echo "Prepares the server for OS migration by backing up important files"
        PRE_MIGRATE=True
        ;;
        '--post')
        echo "Prepares the server for normal operation after the OS migration"
        POST_MIGRATE=True
        ;;
        '--dry-run'|'-d')
        DEBUG=True
        ;;
        '--version'|'-v')
        echo $VERSION
        exit
        ;;
        *)
        echo -e "\nUnrecognized option: '${1}'\n"
        show_help
        exit
        ;;
    esac
    shift
done


#------------------------------------------------------------
# Check for privileged user
#------------------------------------------------------------
if [ $DEBUG == False ]; then
    needs_root
fi

#------------------------------------------------------------
# Redirect all stdout and stderr to a log file
#------------------------------------------------------------
function start_logging()
{
    if [ $PRE_MIGRATE == True ]; then
        mkdir -p $STORAGE_DIR
        chmod 0777 $STORAGE_DIR
        LOGFILE=$STORAGE_DIR/pre-migration_report.txt
        exec > >(tee "$LOGFILE") 2>&1
    elif [ $POST_MIGRATE == True ]; then
        mkdir -p $STORAGE_DIR
        chmod 0777 $STORAGE_DIR
        LOGFILE=$STORAGE_DIR/post-migration_report.txt
        exec > >(tee "$LOGFILE") 2>&1
    fi
    report_header
}

#------------------------------------------------------------
# Main Event
#------------------------------------------------------------
if [ $PRE_MIGRATE == True ]; then
    echo "Pre-migration is about to begin"
    [ $DEBUG == True ] && echo " * * This is a dry-run * *"
    continue_query
    prep_backup_location
    start_logging
    pkg_version 'ion-dbreports'
    stop_essential_services
    backup_essential_files
    backup_postgres
    backup_fstab
    time_to_shutdown
    print_success
elif [ $POST_MIGRATE == True ]; then
    echo "Post-migration is about to begin"
    [ $DEBUG == True ] && echo " * * This is a dry-run * *"
    continue_query
    start_logging
    # Generic system setup
    set_timezone
    restore_network
    restore_essential_files
    restore_fstab
    reset_servername
    reinstall_md1200
    # Torrent Suite setup
    ion_apt_url_check
    gridengine_check
    dbreports_version_check
    restore_postgres
    run_migrations
    enable_essential_services
    rename_log_directory
    print_success
    time_to_shutdown
else
    show_help
fi

exit
