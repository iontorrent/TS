#!/usr/bin/env python
# Copyright (C) 2016 Ion Torrent Systems, Inc. All Rights Reserved

import apt
import os
import sys


def maintenance_mode(action='check'):
    ''' Enable/disable or check status of website maintenance mode '''
    MAINTENANCE_WEB="/var/www/maintenance.enable"
    maintenance = os.path.exists(MAINTENANCE_WEB)

    if action == 'enable' and not maintenance:
        open(MAINTENANCE_WEB, 'a').close()
    elif action == 'disable' and maintenance:
        os.remove(MAINTENANCE_WEB)

    print "ON" if os.path.exists(MAINTENANCE_WEB) else "OFF"


def install_ion_package(name, version):
    ''' Update existing ion package to version '''
    package_list = ['ion-chefupdates']
    if name not in package_list:
        print '%s is not in allowed package list' % name
        sys.exit(1)

    try:
        cache = apt.Cache()
        package = cache[name]
        install_version = package.versions.get(version)
        if not install_version:
            print 'Incorrect package version for %s: %s.' % (name, version)
            sys.exit(1)

        package.candidate = install_version
        package.mark_install()
        cache.commit()
    except SystemError, apt.cache.LockFailedException:
        print 'Unable to install, another update process may be running. Please try again at a later time.'
        sys.exit(1)
    except Exception as err:
        print err
        sys.exit(1)


def check_write_permission(backup_directory):
    '''Check the given directory for write permission'''
    import errno
    import tempfile
    try:
        status = tempfile.NamedTemporaryFile(dir=backup_directory)
        status.close()
    except Exception as e:
        if e.errno in [errno.EPERM, errno.EACCES]:  # Operation not permitted
            errmsg = "Insufficient write permission in %s" % backup_directory
        else:
            errmsg = e
        sys.stderr.write(errmsg + "\n")
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    ''' Usage: sudo /opt/ion/iondb/bin/sudo_utils.py function_name arg1 arg2 ...
    '''
    if len(sys.argv) < 2:
        print "Error: missing function name"
        sys.exit(1)

    func = sys.argv[1]
    if len(sys.argv) > 2:
        func += "('" + "','".join(sys.argv[2:]) + "')"
    else:
        func += "()"

    eval(func)
