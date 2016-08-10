#!/bin/bash
# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.

#Make sure the ion packages are in /root/pkgs

set -x
if [ "$1" != "" ]; then
  DISK=/dev/vdc
  BOOTDIR=/mnt/nd
  /sbin/parted --script ${DISK} mklabel gpt
  DRV_SIZE_512=`blockdev --getsz ${DISK}`
  DRV_END=`expr ${DRV_SIZE_512} \* 512 / 1000000`
  echo ${DRV_SIZE_512} ${DRV_END}
  /sbin/parted -a minimal ${DISK} mkpart Boot ext4 0 24
  /sbin/parted --script ${DISK} mkpart Main ext4 24 ${DRV_END}
  /sbin/parted --script ${DISK} set 1 boot on
  /sbin/parted --script ${DISK} set 1 bios_grub on
  mkfs.ext4 ${DISK}2
  e2label ${DISK}2 TS_VM
  mkdir ${BOOTDIR}
  mount ${DISK}2 ${BOOTDIR}
  MOUNTED=`cat /proc/mounts | grep "${BOOTDIR}"`

  # clean up the mnt directory
  rm -rf /mnt/ionFirstLog
  rm -rf /mnt/.ion*

  if [ "${MOUNTED}" != "" ]; then
    mkdir ${BOOTDIR}/sys
    mkdir ${BOOTDIR}/proc
    mkdir ${BOOTDIR}/mnt
    mkdir ${BOOTDIR}/mnt/external
    mkdir ${BOOTDIR}/media
    cd /; DIRS=`ls -1 | grep -v -e sys -e proc -e mnt -e media -e lost -e 'ion-data'`
    cd /; cp -rp $DIRS ${BOOTDIR}
    echo installing grub
    grub-install --no-floppy ${DISK} --root-directory=${BOOTDIR}
#    mount --bind /dev  ${BOOTDIR}/dev
#    mount --bind /sys  ${BOOTDIR}/sys
#    mount --bind /proc ${BOOTDIR}/proc
#    chroot ${BOOTDIR} update-grub
    UUID_OLD=`cat ${BOOTDIR}/boot/grub/grub.cfg | grep -m 1 UUID | awk -FUUID= ' { print$2 } ' | awk ' { print $1 } '`
    UUID=`blkid | grep ${DISK} | awk -FUUID= ' { print $2 } ' | awk -F\" ' { print $2 } '`
    echo "Old UUID: ${UUID_OLD}  New: ${UUID}"
    if [ ${UUID_OLD} != "" ]; then
      sed -i "s/${UUID_OLD}/${UUID}/g" ${BOOTDIR}/boot/grub/grub.cfg
    fi
    echo done creating disk
  fi
fi

