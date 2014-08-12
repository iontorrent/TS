#!/usr/bin/env python
# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
import os
import subprocess
import json
from celery import task, group
from ion.utils.TSversion import findVersions
from iondb.rundb.models import Cruncher

USER = "ionadmin"

def connect_nodetest(node):
    # Runs /usr/sbin/grp_connect_nodetest to test node connection
    # node status on any failure is 'error'
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
    # Runs /usr/sbin/grp_config_nodetest to test node configurations
    # node status on any failure is 'warning'
    def parse_to_dict(result):
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
    
                status_dict['version_test_errors'] = '\n'.join(['%s %s (%s)'%(v[0],v[1],v[2]) for v in sorted(version_test_errors)])
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


@task
def test_node_and_update_db(node, head_versions):
    # run tests
    node_status = connect_nodetest(node.name)
    if node_status['status'] == 'good':
        node_status.update(config_nodetest(node.name, head_versions))
    
    # update cruncher database entry
    node.state = node_status['status'][0].upper()
    node.info = node_status
    node.save()
    
    return node_status['status']
    
def run_nodetests():
    # Loops over Crunchers and runs node tests on each
    # updates cruncher state and test results in database 
    
    nodes = Cruncher.objects.all().order_by('pk')
    if not nodes:
        return False
    
    head_versions, meta = findVersions()
    # launch parallel celery tasks to test all nodes
    tasks = group(test_node_and_update_db.s(node, head_versions) for node in nodes)()
    node_states = tasks.get()
    
    if 'error' in  node_states:
        cluster_status = 'error'
    elif 'warning' in node_states:
        cluster_status = 'warning'
    else:
        cluster_status = 'good'
        
    return cluster_status
