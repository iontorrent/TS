#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved


"""
This script will do PostgreSQL database migrations, when they are needed.
"""
from __future__ import absolute_import
from djangoinit import *
from django import db
from django.db import transaction, IntegrityError, DatabaseError
import sys

from django.core import management
from south.exceptions import NoMigrations


def fix_south_issues(log):
    # Torrent Server Version 2.2 is hereby declared as db schema version 0001
    # This legacy migration script is still used to get to initial 2.2 schema
    # We must fake inject existing db structures, or South will try to do it too
    # Note initial non-fake attempt for initial installs.
    log.write("Fixing common south issues with and tastypie\n")

    # tastypie started using migrations in 0.9.11
    # So we may have the initial tables already
    try:
        management.call_command('migrate', 'tastypie', verbosity=0, ignore_ghosts=True)
    except DatabaseError as e:
        if "already exists" in str(e):
            management.call_command('migrate', 'tastypie', '0001', fake=True)
        else:
            raise
    except NoMigrations:
        log.write("ERROR: tastypie should have been upgraded to 0.9.11 before now...\n")

    return


def has_south_init():
    try:
        cursor = db.connection.cursor()
        cursor.execute("""SELECT * FROM south_migrationhistory WHERE app_name='rundb' AND migration='0001_initial' LIMIT 1""")
        found_init = (cursor.rowcount == 1)
        cursor.close()
        return found_init
    except (IntegrityError, DatabaseError):
        try:
            transaction.rollback()  # restore django db to usable state
        except transaction.TransactionManagementError:
            pass
    return False

if __name__ == '__main__':

    if has_south_init():
        # south handles this from now on.
        # Ensure south is sane and let it handle the rest.
        fix_south_issues(sys.stdout)


