#!/bin/bash
# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

# eth3 needs to support a secondary Torrent Server
# eth3 needs to change its network address so it doesn't cause the secondary to get a conflicting
#	address on eth0.  (eth3 and eth0 on secondary would be same subnet.)
# iptables and dhcp need to be aware of the subnet change

#iptables
# iptables need to know about 192.168.206.0
if ! grep "-A POSTROUTING -s 192.168.206.0" /etc/iptables.rules; then
	
    if grep "POSTROUTING -s 192.168.204.0" /etc/iptables.rules; then
    	sed -i "s:\(-A POSTROUTING -s 192.168.204.0/24 -o eth0 -j MASQUERADE\):\1\n-A POSTROUTING -s 192.168.206.0/24 -o eth0 -j MASQUERADE:" /etc/iptables.rules
    else
    	sed -i "s:\(-A POSTROUTING -s 192.168.203.0/24 -o eth0 -j MASQUERADE\):\1\n-A POSTROUTING -s 192.168.206.0/24 -o eth0 -j MASQUERADE:" /etc/iptables.rules
    fi
    
    iptables-restore < /etc/iptables.rules
fi

#dhcp
# change the *.*.203.* network to 206
sed -i "s/\.203\./\.206\./g" /etc/dhcp3/dhcpd.conf

#Note: For now, we support only a single secondary TS - until dynamic DNS is enabled
## dhcp server needs to have more addresses to lease
#sed -i "s/range 192.168.206.10.*/range 192.168.206.10 192.168.206.20;/" /etc/dhcp3/dhcpd.conf

#interfaces
ifdown eth3
sed -i "s/\.203\.1/\.206\.1/g" /etc/network/interfaces
ifup eth3

invoke-rc.d dhcp3-server restart

# Mark this node as a Primary Node in a TSNetwork
if [ -w /etc/torrentserver/tsconf.conf ]; then
	sed -i "s/^configuration:.*/configuration:tsnetwork/" /etc/torrentserver/tsconf.conf
else
	echo "Cannot find /etc/torrentserver/tsconf.conf"
    echo "Why not?  Where'd it go?"
fi


exit 0
