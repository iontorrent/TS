#!/bin/bash
# Determines which port is connected to an outbound network
function outbound_net_port()
{
    port=$(/sbin/route | grep ^default | awk '{print $8}')
    if [[ ! -z $port ]]; then
        echo $port
    fi
}

# Checks if an ethernet port exists
function valid_port ()
{
    grep -q $1 /proc/net/dev
    valid=$?
    return $valid
}

function config_PGM_interface()
{

    #--- Remove any previous definitions for interfaces ---#
    sed -i '/auto eth1/,/netmask 255.255.255.0/d' /etc/network/interfaces
    sed -i '/auto eth2/,/netmask 255.255.255.0/d' /etc/network/interfaces
    sed -i '/auto eth3/,/netmask 255.255.255.0/d' /etc/network/interfaces
    sed -i '/auto eth4/,/netmask 255.255.255.0/d' /etc/network/interfaces
    sed -i '/auto eth5/,/netmask 255.255.255.0/d' /etc/network/interfaces

    #--- delete all consecutive blank lines, allows 0 blanks at top, 1 at EOF ---#
    sed -i '/./,/^$/!d' /etc/network/interfaces

#--- Add interface definitions ---#
#
# Here, we want Head Node subnets to be .201, .202, .203, .204
#
interfaces=( 1 2 3 4 5 )
for i in ${interfaces[@]}; do
    if (valid_port eth${i}) && (eth${i} != $(outbound_net_port)); then

        cat >> /etc/network/interfaces <<EOFNIC

auto eth${i}
iface eth${i} inet static
address 192.168.20${i}.1
netmask 255.255.255.0
EOFNIC

    else
        echo "interface eth${i} is invalid. Not configured."
    fi

done

    #--- Enable packet forwarding ---#
    sed -i 's/^#net.ipv4.ip_forward/net.ipv4.ip_forward/' /etc/sysctl.conf
    sed -i 's/^net.ipv4.ip_forward.*/net.ipv4.ip_forward=1/' /etc/sysctl.conf

    sysctl -p


    return 0
}


config_PGM_interface

exit
