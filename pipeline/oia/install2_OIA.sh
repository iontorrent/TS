#!/bin/bash
#/* Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved */

OIA_VERSION=OIAVER  # Version = ANALYSISVER (GITHASH) GCC GCCVER

GPU_PKG_VERSION=GPUVER
ANALYSIS_PKG_VERSION=ANALYSISVER

PKG_LIST=(
ANALYSIS_PKG_NAME
)

# stop OIA
/etc/init.d/oia stop
pkill justBeadFind
pkill Analysis
pkill BaseCaller

mkdir /software/oia
mv oiad.py /software/oia
mv oiaTimingPlot.py /software/oia
mv oia.config /software/oia
rm /software/config/oia.config
ln -s /software/oia/oia.config /software/config/oia.config
mv oia /etc/init.d/
update-rc.d oia defaults

if [ -n "$ANALYSIS_PKG_VERSION" ]; then
    # RUO
    # remove ion-analysis ion-gpu packages older than 5.1.5
    INSTALLED_GPU_PKG_VERSION=`dpkg -l | grep ion-gpu | awk '{print $3}'`
    if [ -z "${INSTALLED_GPU_PKG_VERSION}" ];
    then
        INSTALLED_GPU_PKG_VERSION=1.0
    fi
    echo ${INSTALLED_GPU_PKG_VERSION}
    
    INSTALLED_ANALYSIS_PKG_VERSION=`dpkg -l | grep ion-analysis | grep -v ts-ion-analysis | awk '{print $3}'` 
    if [ -z "${INSTALLED_ANALYSIS_PKG_VERSION}" ];
    then
        INSTALLED_ANALYSIS_PKG_VERSION=1.0
    fi
    echo ${INSTALLED_ANALYSIS_PKG_VERSION}
    
    if dpkg --compare-versions ${INSTALLED_GPU_PKG_VERSION} lt 5.1.5 ||
       dpkg --compare-versions ${INSTALLED_ANALYSIS_PKG_VERSION} lt 5.1.5 ;
    then
        dpkg -r ion-analysis ion-gpu
    fi
    
    
    # make sure gpu installs first, Analysis might depend on a specific version
    if [ ! "ii ion-gpu $GPU_PKG_VERSION" = "`dpkg -l | grep ion-gpu | awk '{print $1, $2, $3}'`" ];
    then
        # stop lightdm server
        service lightdm stop
    
        # stop ganglia-monitor server (which might monitor the GPU)
        service ganglia-monitor stop
    
        # install driver
        dpkg -i GPU_PKG_NAME
        # remove xorg.conf (installed by CUDA)
        rm /etc/X11/xorg.conf
    fi
    rm -f GPU_PKG_NAME
    
    # disable ECC support
    nvidia-smi -e 0
    
    dpkg -i --auto-deconfigure "${PKG_LIST[@]}"
    rm -f "${PKG_LIST[@]}"
    
    if [ "ii ion-gpu $GPU_PKG_VERSION"           = "`dpkg -l | grep ion-gpu      | awk '{print $1, $2, $3}'`" ] &&
       [ "ii ion-analysis $ANALYSIS_PKG_VERSION" = "`dpkg -l | grep ion-analysis | grep -v ts-ion-analysis | awk '{print $1, $2, $3}'`" ];
    then
        echo OIA Version: $OIA_VERSION > /software/OIAVersion
    fi
else
    # DX 
    echo OIA Version: $OIA_VERSION > /software/OIAVersion
fi
