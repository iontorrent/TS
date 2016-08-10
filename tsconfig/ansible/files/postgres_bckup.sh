#!/bin/bash
# Copyright (C) 2015 Thermo Fisher Scientific. All Rights Reserved.
# Make a backup of the database.  This is intended to backup an existing TS
# prior to installing ion-dbreports which usually makes db changes.
BACKUP_DBASE_DIR=/results
if [ -d $BACKUP_DBASE_DIR ]; then
#    user_msg "=================================================================="
#    user_msg "\tCreating backup of postgresql database"
#    user_msg "=================================================================="
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DBASE_FILE="${BACKUP_DBASE_DIR}/iondb.${TIMESTAMP}.backup"
    /usr/bin/pg_dump -U ion -c iondb | gzip > "${BACKUP_DBASE_FILE}.gz" || true

    # Keep 5 most recent dbase backup files
    KEEP=5
    filecmd="ls -rt ${BACKUP_DBASE_DIR}/iondb.*.backup.gz"
    FILES=($($filecmd 2>/dev/null || true))
    cnt=${#FILES[@]}
    while [ $cnt -gt $KEEP ]; do
        rm -f ${FILES[0]}
        FILES=($($filecmd 2>/dev/null || true))
        cnt=${#FILES[@]}
    done
fi