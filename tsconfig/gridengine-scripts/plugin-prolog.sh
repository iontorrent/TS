#!/bin/bash
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
## SGE prolog script

# Rotate drmaa_stdout.txt file to numbered backup
if [ -e "${SGE_STDOUT_PATH}" ]; then
    TMPFILE="${SGE_STDOUT_PATH}.new.${JOB_ID}"
    touch ${TMPFILE}
    mv --backup=numbered ${TMPFILE} ${SGE_STDOUT_PATH}
fi

