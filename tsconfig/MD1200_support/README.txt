===============================================================================
MD1200 Support Package for Ion Torrent Server
===============================================================================
Contents:
    driver_update
    README.txt
    stresstest.tgz
    TSaddstorage
    dkms_2.1.1.2-2ubuntu1_all.deb
	libdevmapper-event1.02.1_2%3a1.02.39-1ubuntu4.1_amd64.deb
    lsscsi_0.21-2build1_amd64.deb
	lvm2_2.02.54-1ubuntu4.1_amd64.deb
	megacli_8.04.53-2_all.deb
	megaraid-sas-dkms_00.00.05.39_all.deb
	watershed_5_amd64.deb
	xfsprogs_3.1.0ubuntu1_amd64.deb


Installation instructions:

	For TS version earlier than 4.0:
    
    	Run ./driver_update prior to pwoering down and installing hardware.

	Ensure that the MD1200 enclosure has been attached properly to the Torrent Server.

    Run the configuration utility:

        sudo ./TSaddstorage

===============================================================================
Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
Contact: bernard.puc@lifetech.com
===============================================================================
