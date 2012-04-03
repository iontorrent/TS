#!/bin/sh

# point to drmdev axeda server if on lifetech LAN
host lifelink.corp.life
if [ "$?" = "0" ]
then
  ./RSMAgent_TS https://drmdev.appliedbiosystems.com/eMessage
else
  ./RSMAgent_TS https://drm.appliedbiosystems.com/eMessage
fi

