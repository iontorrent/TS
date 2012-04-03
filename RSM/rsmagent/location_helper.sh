#!/bin/sh

# get my public IP address
#wget http://www.whatismyip.com/automation/n09230945.asp -O ip.txt

# get my location and save into json file
#wgetcmd="http://www.geobytes.com/IpLocator.htm?GetLocation&template=json.txt&ipaddress="`cat ip.txt`
#wget -O loc.txt $wgetcmd

# get my public IP address
wget -q -O - checkip.dyndns.org|sed -e 's/.*Current IP Address: //' -e 's/<.*$//' > ip.txt
# get my location and save into colon-delimited file
whois `cat ip.txt` | grep network | sed "s/network://" > loc.txt

