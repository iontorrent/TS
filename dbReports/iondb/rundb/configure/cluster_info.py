#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
'''Utility functions to ascertain cluster node health'''
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

USER = "ionadmin"


def connect_nodetest(node):
    '''
    # Runs /usr/sbin/grp_connect_nodetest to test node connection
    # node status on any failure is 'error'
    '''
    script = '/usr/sbin/grp_connect_nodetest'
    command = ['sudo', '-u', USER, script, node]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    status_dict = {
        'name': node,
        'status': 'good',
        'connect_tests': [],
        'error': stderr
    }

    for line in stdout.splitlines():
        try:
            test_name, test_result = line.split(':')
            status_dict['connect_tests'].append((test_name, test_result))
            if test_result == 'failure':
                status_dict['status'] = 'error'
        except:
            status_dict['status'] = 'error'
            status_dict['error'] += 'Unable to get status from: ' + line + '\n'

    return status_dict


def config_nodetest(node, head_versions):
    '''
    # Runs /usr/sbin/grp_config_nodetest to test node configurations
    # node status on any failure is 'warning'
    '''
    def parse_to_dict(result):
        '''helper function'''
        result = result.split(',')
        return dict([(v.split('=')[0], v.split('=')[1]) for v in result if len(v.split('=')) == 2])

    script = '/usr/sbin/grp_config_nodetest'
    command = ['sudo', '-u', USER, script, node]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    status_dict = {
        'status': 'good',
        'config_tests': [],
        'error': stderr
    }

    for line in stdout.splitlines():
        try:
            test_name, test_result = line.split(':')
            if test_name == 'version_test':
                # process versions
                versions = parse_to_dict(test_result)
                version_test_status = 'success'
                version_test_errors = []
                for key, version in versions.items():
                    if key in head_versions and version != head_versions[key]:
                        version_test_status = 'failure'
                        version_test_errors.append((key, version, head_versions[key]))
                        status_dict['status'] = 'warning'

                status_dict['version_test_errors'] = '\n'.join(['%s %s (%s)' % (v[0], v[1], v[2]) for v in sorted(version_test_errors)])
                status_dict['config_tests'].append((test_name, version_test_status))
                if version_test_status == 'failure':
                    status_dict['error'] += 'Software versions do not match headnode\n'
            else:
                status_dict['config_tests'].append((test_name, test_result))
                if test_result == 'failure':
                    status_dict['status'] = 'warning'
        except:
            status_dict['status'] = 'warning'
            status_dict['error'] += 'Unable to get status from: ' + line + '\n'

    return status_dict


def queue_info(node=''):
    info = {}
    try:
        command = ["qstat", "-f"]
        if node:
            command.extend(["-q", "*@" + node.strip()])

        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        for line in stdout.splitlines():
            l = line.split()
            if len(l) > 1 and '.q' in l[0]:
                name = os.path.splitext(l[0].split('@')[1])[0]
                if name not in info:
                    info[name] = {
                        'used': 0,
                        'total': 0,
                        'disabled': 0,
                        'error': 0,
                        'load': l[3]
                    }

                resv, used, total = l[2].split('/')
                info[name]['used'] += int(used)
                info[name]['total'] += int(total)

                state = l[5] if len(l) > 5 else ''
                if state == 'E':
                    info[name]['error'] += int(total)
                if 'd' in state:
                    info[name]['disabled'] += int(total)

    except Exception as err:
        logger.error(err)

    return info.get(node,{}) if node else info


def sge_ctrl(action, node):
    ''' Run command and return error '''
    if action == "enable":
        command = "sudo -u %s qmod -e *@%s" % (USER, node)
    elif action == "disable":
        command = "sudo -u %s qmod -d *@%s" % (USER, node)

    error = ''
    try:
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            error = "%s %s" % (stdout, stderr)
    except Exception as e:
        error = str(e)

    return error
