#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import
import os
import subprocess
from itertools import chain
from celery import task
from celery.result import AsyncResult
from celery.utils.log import get_task_logger

@task
def _do_celery_subprocess(command):
    response_obj = {}
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='/results/tsvm')
        stdout, stderr = proc.communicate()
        if proc.returncode:
            response_obj['state'] = 'Error'
            response_obj['msg'] = "subprocess Return code: %d" % (proc.returncode)
    except Exception as e:
        response_obj['state'] = 'Error'
        response_obj['msg'] = e.child_traceback
    else:
        response_obj['state'] = stdout
        response_obj['msg'] = stderr
    return response_obj


def _do_subprocess(command, User=None):
    '''General purpose shell to call tsvm bash scripts'''
    logger = get_task_logger(__name__)

    #Handle explicitly the case where /results/tsvm does not exist.  means ion-tsvm package is not installed
    if not os.path.isdir('/results/tsvm'):
        response_obj = {}
        response_obj['state'] = 'Error'
        response_obj['msg'] = "Directory not found /results/tsvm.  Install ion-tsvm package."
    else:
        #NOTE: we use celery because it is running with root permission on our server.  Thus,
        #calling sudo here will not result in a password prompt.
        if User:#Format the command line to execute as a certain user (instead of root)
            command.insert(0, 'sudo')
            command.insert(1, '-u')
            command.insert(2, User)
        #Execute a celery task that executes the shell script
        mytask = _do_celery_subprocess.delay(command)
        #Wait for celery task to complete
        response_obj = AsyncResult(mytask.task_id).get()

    return response_obj


def status():
    '''Returns the current status of the VM'''
    #First of all, check if Vt is disabled in the BIOS
    #Must be run with super user permissions
    command = ['./tsvm-isvtenabled']
    response_obj = _do_subprocess(command)
    if response_obj['state'] in ['BIOS Disabled', 'Error']:
        return response_obj

    response_obj = {'action': 'status'}
    command = ['./tsvm-status']
    response_obj = dict(chain(response_obj.iteritems(), _do_subprocess(command, User='ionadmin').iteritems()))
    return response_obj


def setup(version):
    '''Sends commands to initialize/setup the VM'''

    # As root user, need to ensure the iptables firewall rules allow NFS export to the TS-VM
    command = ['bash', '-lc', 'source ./tsvm-include; enable_nfs_rule']
    response_obj = _do_subprocess(command)
    if response_obj['state'] == 'Error':
        return response_obj

    response_obj = {'action': 'setup'}
    command = ['./tsvm-setup', '--version', version]
    response_obj = dict(chain(response_obj.iteritems(), _do_subprocess(command, User='ionadmin').iteritems()))
    return response_obj


def ctrl(action):
    '''Sends commands to control the VM'''
    response_obj = {'action': action}
    command = ['./tsvm-ctrl', action]
    response_obj = dict(chain(response_obj.iteritems(), _do_subprocess(command, User='ionadmin').iteritems()))
    return response_obj


def versions():
    '''Returns array of valid TS versions available'''
    command = ['./tsvm-versions']
    response_obj = _do_subprocess(command, User='ionadmin')
    if 'Error' in response_obj['state']:
        versions = []
    else:
        versions = response_obj['state']
    return versions

@task
def check_for_new_tsvm():
    '''Check if there is an updated version of ion-tsvm package'''
    # NOTE: Must be run with root privileges
    response_obj = {'action': 'check_update'}
    try:
        import pkg_resources
        from distutils.version import LooseVersion
        import apt
        apt_cache = apt.Cache()
        apt_cache.update()
        apt_cache.open()
        pkg = apt_cache['ion-tsvm']

        if LooseVersion(pkg_resources.get_distribution("python-apt").version) >= LooseVersion('0.9.3.5'):
            if bool(pkg.is_installed):
                state = 'Upgradable' if bool(pkg.is_upgradable) else 'NotUpgradable'
            else:
                state = 'NotInstalled'

            response_obj.update({
                'state': state,
                'msg': 'New ion-tsvm package version %s is available' % pkg.candidate
            })
        else:
            # compatible with python-apt 0.7.94.1 on ubuntu 10.04 - Lucid
            if bool(pkg.isInstalled):
                state = 'Upgradable' if bool(pkg.isUpgradable) else 'NotUpgradable'
            else:
                state = 'NotInstalled'

            response_obj.update({
                'state': state,
                'msg': 'New ion-tsvm package version %s is available' % pkg.candidateVersion
            })

    except Exception as e:
        response_obj.update({'state': 'Error', 'msg': str(e)})

    return response_obj

@task
def install_new_tsvm():
    '''Installs new ion-tsvm package'''
    # NOTE: Must be run with root privileges
    response_obj = {'action': 'update'}
    try:
        import pkg_resources
        from distutils.version import LooseVersion
        import apt
        apt_cache = apt.Cache()
        apt_cache.update()
        apt_cache.open()
        pkg = apt_cache['ion-tsvm']

        if LooseVersion(pkg_resources.get_distribution("python-apt").version) >= LooseVersion('0.9.3.5'):
            if bool(pkg.is_installed):
                pkg.mark_upgrade()
            else:
                pkg.mark_install()
        else:
            # compatible with python-apt 0.7.94.1 on ubuntu 10.04 - Lucid
            if bool(pkg.isInstalled):
                pkg.markUpgrade()
            else:
                pkg.markInstall()

        apt_cache.commit()
        response_obj.update({'state':'Success', 'msg': "ion-tsvm updated"})
    except Exception as e:
        response_obj.update({'state': 'Error', 'msg': str(e) })

    return response_obj
