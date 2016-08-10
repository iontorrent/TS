#!/bin/bash
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved
# Purpose: update existing all_local with new settings
set -e
target=/usr/share/ion-tsconfig/ansible/group_vars/all_local
target=./all_local
[[ ! -f "$target" ]] && exit

#=============================================================================
# Edits for TSS5.2 release
#=============================================================================
# postgres_sysctl_file: filename change
sed -i "s,^postgres_sysctl_file.*,postgres_sysctl_file: /etc/sysctl.d/30-postgresql-shm.conf," "$target"

# ion_apt_server: new variable defined
if ! grep -q ^ion_apt_server "$target"; then
    line="ion_apt_server: ionupdates.com"
    # insert this line above line starting with 'tsconfig_dir'
    sed -i "/^tsconfig_dir.*$/i $line\n" "$target"
fi

# iru_apt_server: new variable defined
if ! grep -q ^iru_apt_server "$target"; then
    line="iru_apt_server: iru.ionreporter.thermofisher.com"
    # insert this line above line starting with 'tsconfig_dir'
    sed -i "/^tsconfig_dir.*$/i $line\n" "$target"
fi

# new network variable: dns_search
if ! grep -q ^dns_search "$target"; then
    line="dns_search:"
    # insert this line after line starting with dns_nameserver
    sed -i "/^dns_nameserver.*/a $line" "$target"
fi

# new variable: config_firewall
if ! grep -q ^config_firewall "$target"; then
    line="config_firewall: True"
    # insert this line after line starting with dns_search
    sed -i "s/\(^dns_search.*\)/\1\n\n$line/" "$target"
fi
# new variable: enable_dhcp
if ! grep -q ^enable_dhcp "$target"; then
    line="enable_dhcp: True"
    # insert this line after line starting with dns_search
    sed -i "s/\(^dns_search.*\)/\1\n\n$line/" "$target"
fi
# new variable: enable_hosts_copy
if ! grep -q ^enable_hosts_copy "$target"; then
    line="enable_hosts_copy: False"
    # insert this line after line starting with dns_search
    sed -i "s/\(^dns_search.*\)/\1\n\n$line/" "$target"
fi
# new variable: edit_interfaces
if ! grep -q ^edit_interfaces "$target"; then
    line="edit_interfaces: True"
    # insert this line after line starting with dns_search
    sed -i "s/\(^dns_search.*\)/\1\n\n$line/" "$target"
fi

# new variable: UPDATE_SYSTEM
if ! grep -q ^UPDATE_SYSTEM "$target"; then
    line="UPDATE_SYSTEM: True"
    # insert line above end-of-customizable section
    lineno=$(grep -n "End of user customizable section" "$target"|awk -F: '{print $1}')
    lineno=$((lineno-1))
    sed -i "${lineno}i $line\n" "$target"
fi
