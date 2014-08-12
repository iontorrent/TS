#!/bin/bash
function config_dhcp()
{
    echo "Configuring DHCP Server"
#--- ENABLE DNS SERVER ---#
#---                                                                        ---#
#---    Use DNS provided by dnsmasq; but turn off DHCP provided by dnsmasq  ---#
#---                                                                        ---#
    #--- Delete any existing config file ---#
    if [ -w /etc/dnsmasq.d/ion-dhcp ]; then
        rm -f /etc/dnsmasq.d/ion-dhcp
    fi

    echo "no-hosts" > /etc/dnsmasq.d/ion-dhcp
    echo "addn-hosts=/etc/hosts-dnsmasq" >> /etc/dnsmasq.d/ion-dhcp
    echo "no-dhcp-interface=eth0" >> /etc/dnsmasq.d/ion-dhcp
    interfaces=( 1 2 3 4 5 )
    for i in ${interfaces[@]}; do
        if valid_port eth${i}; then echo -e "interface=eth${i}\nno-dhcp-interface=eth${i}\n" >> /etc/dnsmasq.d/ion-dhcp; fi
    done
    #--- Remove any pre-existing entries in /etc/hosts ---#
    #---    Cleans out older TSconfigged hosts files
    sed -i '/^192.168.1.1.*/d' /etc/hosts
    sed -i '/^192.168.201.10.*/d' /etc/hosts
    sed -i '/^192.168.202.10.*/d' /etc/hosts
    sed -i '/^192.168.203.10.*/d' /etc/hosts
    sed -i '/^192.168.204.10.*/d' /etc/hosts

    #---    Overwrite existing hosts-dnsmasq    ---#
    echo "# $(date)" > /etc/hosts-dnsmasq
    echo -e "192.168.1.1\tts" >> /etc/hosts-dnsmasq
    interfaces=(1 2 3 4 5)
    for i in ${interfaces[@]}; do
        if valid_port eth${i}; then echo -e "192.168.20${i}.10\tpgm${i}" >> /etc/hosts-dnsmasq;fi
    done

    #---    Enable reading of dnsmasq config file   ---#
    sed -i 's:^#conf-dir.*:conf-dir=/etc/dnsmasq.d:' /etc/dnsmasq.conf
    user_msg "See /etc/dnsmasq.conf"

#--- ENABLE DHCP SERVER ---#

    #--- Backup old config file ---#
    if [ -w /etc/dhcp3/dhcpd.conf ]; then
        mv /etc/dhcp3/dhcpd.conf /etc/dhcp3/dhcpd.conf.old
    fi

    #---                            ---#
    #--- Write out new config file  ---#
    #---                            ---#
cat >> /etc/dhcp3/dhcpd.conf <<EOFDHCP
ddns-update-style none;
default-lease-time 300;
max-lease-time 300;
log-facility local7;
option domain-name "pgm.local";
EOFDHCP

interfaces=( 1 2 3 4 5 )
for i in ${interfaces[@]}; do
    if valid_port eth$i; then

        cat >> /etc/dhcp3/dhcpd.conf <<EOFDHCP
# Ion PGM subnet
subnet 192.168.20$i.0 netmask 255.255.255.0 {
    range 192.168.20$i.10 192.168.20$i.10;
    option subnet-mask  255.255.255.0;
    option broadcast-address 192.168.20$i.255;
    option routers 192.168.20$i.1;
    option domain-name-servers 192.168.20$i.1;
    option netbios-name-servers 192.168.20$i.1;
}
EOFDHCP

    else
        echo "dhcp config: port eth$i invalid, not configuring" 1>&2
    fi
done



    #---    Edit dhclient.conf          ---#
    if [ -e /etc/dhcp3/dhclient.conf ]; then
        mv /etc/dhcp3/dhclient.conf /etc/dhcp3/dhclient.conf.orig
    fi

    cat >> /etc/dhcp3/dhclient.conf <<EOFDHCLIENT
option rfc3442-classless-static-routes code 121 = array of unsigned integer 8;

send host-name "<hostname>";

interface "eth0"
{
request subnet-mask, broadcast-address, time-offset, routers,
        domain-name, domain-name-servers, domain-search, host-name,
        netbios-name-servers, netbios-scope, interface-mtu,
        rfc3442-classless-static-routes, ntp-servers;
}
EOFDHCLIENT

    #TODO: Enable proper error checking on return code here
    service dhcp3-server restart || true

    return 0
}

config_dhcp

exit
