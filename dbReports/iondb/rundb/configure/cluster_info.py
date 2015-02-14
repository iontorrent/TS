#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
'''Utility functions to ascertain cluster node health'''
import subprocess

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
        return dict([(v.split('=')[0], v.split('=')[1]) for v in result if len(v.split('='))==2])

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
    
                status_dict['version_test_errors'] = '\n'.join(['%s %s (%s)'%(v[0], v[1], v[2]) for v in sorted(version_test_errors)])
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
