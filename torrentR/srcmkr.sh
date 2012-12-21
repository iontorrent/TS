#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
###--------------------------------------------------------------------------###
###	srcmkr.sh
###	Create tar archive of source files for Ion Software packages
###	This script is intended to be called from ./buildTools/build.sh and
###	executed from the build/$MODULE directory.
###
###--------------------------------------------------------------------------###

absPath=$(readlink -f $(which $0))
absPath=$(dirname $absPath)
PKG=$(basename $absPath)
cd $absPath/..

#---	Name the tarball with version info	---#
source ${absPath}/version
SRC_FILENAME="ion-${PKG}_src_${MAJOR}.${MINOR}.${RELEASE}" 
DST_PATH="build/$PKG/$SRC_FILENAME"

#---	Feel-good output	---#
echo "Creating source archive for: "$SRC_FILENAME

#---	Create staging directory	---#
mkdir -p ${DST_PATH}

#---	Copy source files to staging area	---#
cp -rp ./$PKG ${DST_PATH}/

mkdir -p ${DST_PATH}/buildTools
cp -p ./buildTools/build.sh ${DST_PATH}/buildTools
cp -rp ./buildTools/cmake ${DST_PATH}/buildTools/
cp -rp ./buildTools/IonVersion.h.in ${DST_PATH}/buildTools/
cp -rp ./buildTools/IonVersion.cpp.in ${DST_PATH}/buildTools/
cp -rp ./buildTools/IonVersion.env.in ${DST_PATH}/buildTools/
cp -rp ./buildTools/LICENSE.txt.in ${DST_PATH}/buildTools/
cp -rp ./buildTools/BUILD.txt ${DST_PATH}/buildTools/
cp -rp ./buildTools/dbgmem.* ${DST_PATH}/buildTools/
cp -rp ./buildTools/cleanTorrentR.sh ${DST_PATH}/buildTools/

mkdir -p ${DST_PATH}/external
cp -rp ./external/samtools-0.1.18 ${DST_PATH}/external/
cp -rp ./external/bamtools-2.1.1.20120920+gite235c55 ${DST_PATH}/external/

#---	Purge svn control files	---#
find ${DST_PATH} -name .svn\* -exec rm -rf {} \; 2>/dev/null
find ${DST_PATH} -name ChangeLog -exec rm -rf {} \; 2>/dev/null
rm ${DST_PATH}/${PKG}/srcmkr.sh

#---	Create top level guidance files	---#
echo "Please see buildTools/LICENSE.txt.in for license information" > ${DST_PATH}/LICENSE.txt
echo -e \
"Please see buildTools/BUILD.txt for build requirements and instructions.\n"\
"Note especially the section BUILD SPECIFIC MODULES.\n"\
"To build Analysis module, the command is:\n"\
"MODULES=Analysis ./buildTools/build.sh\n"\
> ${DST_PATH}/README.txt

#---	Tar archive source files	---#
cd build/$PKG
tar zcf ${SRC_FILENAME}.tgz ${SRC_FILENAME}

#---	Clean up	---#
rm -rf ${SRC_FILENAME}

#---	Move from volatile ./build subdirectory	---#
mv ${SRC_FILENAME}.tgz ${absPath}/../

exit 0
