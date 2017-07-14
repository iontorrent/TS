#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import

import os
import subprocess

# generate a list of directories and script paths
tsvm_directory = os.path.join('/', 'results', 'tsvm')
tsvm_versions_script = os.path.join(tsvm_directory, 'tsvm-versions')
tsvm_isvtenabled = os.path.join(tsvm_directory, 'tsvm-isvtenabled')
tsvm_status = os.path.join(tsvm_directory, 'tsvm-status')
tsvm_setup = os.path.join(tsvm_directory, 'tsvm-setup')


def status():
    """
    Returns the current status of the VM
    """

    # check to see that the scripts are available
    if not os.path.exists(tsvm_isvtenabled):
        return {
            'action': 'status',
            'state': 'Error',
            'msg': 'Cannot find ' + tsvm_isvtenabled +'. Try installing ion-tsvm?'
        }
    if not os.path.exists(tsvm_status):
        return {
            'action': 'status',
            'state': 'Error',
            'msg': 'Cannot find ' + tsvm_status + '. Try installing ion-tsvm?'
        }

    # Check for BIOS virtualization enabled
    # Must be run with super user permissions
    isvtenabled = subprocess.Popen(['sudo', tsvm_isvtenabled], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = isvtenabled.communicate()
    if stderr or stdout in ['BIOS Disabled', 'Error']:
        return {
            'action': 'status',
            'state': 'Error',
            'msg': stderr
        }

    # get the status
    try:
        return {
            'action': 'status',
            'state': 'status',
            'msg': subprocess.check_output(tsvm_status, cwd=tsvm_directory)
        }
    except subprocess.CalledProcessError as exc:
        return {
            'action': 'status',
            'state': str(exc),
            'msg': 'Error'
        }


def setup(version):
    """
    Sends commands to initialize/setup the VM
    :param version: The version to setup
    :return: A dictionary of the response object
    """

    try:
        return {
            'action': 'setup',
            'msg': subprocess.check_output([tsvm_setup, "--version", str(version)], cwd=tsvm_directory),
            'state': 'setup'
        }
    except subprocess.CalledProcessError as exc:
        return {
            'action': 'setup',
            'state': 'Error',
            'msg': str(exc) + " | " + str(exc.output)
        }
    except OSError as exc:
        return {
            'action': 'setup',
            'state': 'Error',
            'msg': str(exc)
        }


def ctrl(action):
    """
    Sends commands to control the VM
    :param action: The action to execute on the vm
    :return: a dictionary or the response items
    """

    try:
        return {
            'action' : action,
            'state' : subprocess.check_output(['./tsvm-ctrl', action], cwd=tsvm_directory),
            'msg' : ''
        }
    except subprocess.CalledProcessError as exc:
        return {
            'action' : action,
            'state' : 'Error',
            'msg' : str(exc)
        }


def versions():
    """
    Returns array of valid TS versions available
    :return: A list of all of the available TSVM versions
    """

    # if the script does not exist then we just return an empty list
    if not os.path.exists(tsvm_versions_script):
        return list()

    try:
        # execute the scipt and return the array of versions
        response_str = subprocess.check_output(tsvm_versions_script, cwd=tsvm_directory)
        return response_str.strip().split(' ')
    except subprocess.CalledProcessError:
        return list()