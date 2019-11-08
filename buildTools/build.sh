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
  django
  pipeline
  publishers
  tsconfig
"}

PROJECT_ROOT=`pwd`

if [ -z ${BUILD_DIR} ]; then
    BUILD_DIR=`pwd`/build
fi

for M in ${MODULES}; do
  if [ ! -d "$M" ]; then
    echo "Must run $0 from the root folder which has the following folders:"
    for MM in ${MODULES}; do
      if [ -d "$MM" ]; then
        echo " - $MM"
      else
        echo " - $MM (not found)"
      fi
    done
    exit -1;
  fi
done

# set number of jobs
DEFAULT_JOB_NUM=13
if `which nproc &> /dev/null`; then
    # not to overload less capable system
    num_jobs=$(( $(nproc)+1 ))
else
    num_jobs=$DEFAULT_JOB_NUM
fi

# only limit to the original number
if [[ $num_jobs -gt $DEFAULT_JOB_NUM ]]; then
    num_jobs=$DEFAULT_JOB_NUM
fi

# == DX Settings ==
# ANALYSIS_DIR='/server/share/common/analysis'

cmake_opts="-DION_AVX:BOOL=FALSE"
ANALYSIS_DIR=${ANALYSIS_DIR-""}
if [ ! -z $ANALYSIS_DIR ]; then
  cmake_opts+=" -DCMAKE_INSTALL_PREFIX=${ANALYSIS_DIR}"
  cmake_opts+=" -DION_INSTALL_PREFIX=${ANALYSIS_DIR}"
  cmake_opts+=" -DION_PICARD_PREFIX=${ANALYSIS_DIR}/picard"
  cmake_opts+=" -DION_HTML_PREFIX=${ANALYSIS_DIR}/var/www"  
fi

ERR=0
ERRMSG=""
for MODULE in ${MODULES}; do
  MODULE_SRC_PATH=${PROJECT_ROOT}/${MODULE}
  MODULE_BUILD_PATH=${BUILD_DIR}/${MODULE}
  echo "=================================================="
  echo " Building module $MODULE_BUILD_PATH"
  echo "=================================================="
  mkdir -p ${MODULE_BUILD_PATH}
  (
    LOCALERR=0
    find ${MODULE_BUILD_PATH} -name \*.deb | xargs rm -f
    cd ${MODULE_BUILD_PATH}
    cmake_cmd="cmake $@ -G 'Unix Makefiles' ${MODULE_SRC_PATH} ${cmake_opts}"
    echo $cmake_cmd
    eval $cmake_cmd
  
    if [ "$?" != 0 ]; then LOCALERR=1; fi
      if [ "$MODULE" = "rndplugins" ]; then
        make
      else
        make -j $num_jobs
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
  if [ ${MODULE} = "plugin" -o ${MODULE} = "rndplugins" ]; then
    ${PROJECT_ROOT}/buildTools/removeAboutFile.sh ${PROJECT_ROOT}/${MODULE}
  fi
done;

# if the environmental variable is set to create a repository, we will move all of the packages into that repository and
# index them so aptitude can read them in as a repository
if [ ! -z ${MAKE_REPO_DIRECTORY} ]; then
  mkdir -p ${BUILD_DIR}/repo
  find ${BUILD_DIR} -type f -iname "*.deb" ! -path "$BUILD_DIR/repo/*" -exec mv {} ${BUILD_DIR}/repo \;
  cd ${BUILD_DIR}/repo && dpkg-scanpackages -m ./ > ${BUILD_DIR}/repo/Packages
fi

if [ ${ERR} != 0 ]; then
  echo -e ${ERRMSG}
  echo "FAILURES: $ERR modules failed to build."
  exit ${ERR}
fi
echo "SUCCESS: All modules built."

