#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

# Version of build is now
# determined from "version" file
# in each module folder by the
# cmake build system

MODULES=${MODULES-"
  gpu
  Analysis
  torrentR
  torrentPy
  dbReports
  pipeline
  publishers
  tsconfig
  RSM
  tsvm
"}

BUILD_ROOT=`pwd`

for M in $MODULES; do
  if [ ! -d "$M" ]; then
    echo "Must run $0 from the root folder which has the following folders:"
    for MM in $MODULES; do
      if [ -d "$MM" ]; then
        echo " - $MM"
      else
        echo " - $MM (not found)"
      fi
    done
    exit -1;
  fi
done

ERR=0
ERRMSG=""
for MODULE in $MODULES; do
  echo "=================================================="
  echo " Building module $MODULE"
  echo "=================================================="
  mkdir -p build/$MODULE
  (
    LOCALERR=0
    find build/$MODULE -name \*.deb | xargs rm -f
    cd build/$MODULE
    cmake $@ -G 'Unix Makefiles' $BUILD_ROOT/$MODULE -DION_AVX:BOOL=FALSE
    if [ "$?" != 0 ]; then LOCALERR=1; fi
      if [ "$MODULE" = "rndplugins" ]; then
        make
      else
        make -j13
      fi
    if [ "$?" != 0 ]; then LOCALERR=1; fi
      make test
    if [ "$?" != 0 ]; then LOCALERR=1; fi

    make package

    if [ "$?" != 0 ]; then LOCALERR=1; fi
    find . -name _CPack_Packages | xargs rm -rf
# do not delete; only used for official builds
#    if [ -x ../../$MODULE/srcmkr.sh ]; then
#      ../../$MODULE/srcmkr.sh
#    fi
    if [ "$LOCALERR" != 0 ]; then
      false
    else
      true
    fi
  )
  if [ "$?" != 0 ]; then
    ERR=$(($ERR + 1))
    ERRMSG="${ERRMSG}Build of module $MODULE failed.\n"
  fi
  echo "=================================================="
  echo
  if [ $MODULE = "plugin" -o $MODULE = "rndplugins" ]; then
    $BUILD_ROOT/buildTools/removeAboutFile.sh $BUILD_ROOT/$MODULE
  fi
done;

# if the environmental variable is set to create a repository, we will move all of the packages into that repository and
# index them so aptitude can read them in as a repository
if [ ! -z ${MAKE_REPO_DIRECTORY} ]; then
  mkdir -p ${BUILD_ROOT}/build/repo
  find ${BUILD_ROOT}/build -type f -iname "*.deb" ! -path "$BUILD_ROOT/build/*" -exec mv {} ${BUILD_ROOT}/build/repo \;
  cd ${BUILD_ROOT}/build/repo && dpkg-scanpackages -m ./ > ${BUILD_ROOT}/build/repo/Packages
fi

if [ $ERR != 0 ]; then
  echo -e $ERRMSG
  echo "FAILURES: $ERR modules failed to build."
  exit $ERR
else
  echo "SUCCESS: All modules built."
fi

