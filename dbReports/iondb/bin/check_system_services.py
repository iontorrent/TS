#!/usr/bin/python
# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

'''
    This script can test essential TS system services are running and restart them
    Usage:
        no option:      print status
        -n/--notify:    update webpage banner and send message to IT contact if any service is down
        -s/--start:     attempt to (re-)start all services, must be run as root
'''

import os
import sys
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'

import subprocess
import traceback
import argparse
from iondb.rundb.models import Message
from iondb.rundb.tasks import notify_services_error

import logging
logger = logging.getLogger(__name__)


def process_status():
    # modified from /rundb/configure/views.py process_set()
    def simple_status(name):
        proc = subprocess.Popen("service %s status" % name, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        #logger.debug("%s out = '%s' err = %s''" % (name, stdout, stderr))
        return proc.returncode == 0
    
    def complicated_status(filename, parse):
        try:
            if os.path.exists(filename):
                data = open(filename).read()
                pid = parse(data)
                proc = subprocess.Popen("ps %d" % pid, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc.communicate()
                return proc.returncode == 0
        except Exception as err:
            return False

    proc_set = {}
    processes = [
        "ionJobServer",
        "ionCrawler",
        "ionPlugin",
        "apache2",
        "postgresql"
    ]

    for name in processes:
        proc_set[name] = simple_status(name)

    proc_set["RabbitMQ"] = complicated_status("/var/run/rabbitmq/pid", int)
    proc_set["gridengine"] = complicated_status("/var/run/gridengine/execd.pid", int)

    for node in ['celerybeat', 'celery_w1', 'celery_plugins', 'celery_periodic', 'celery_slowlane', 'celery_transfer', 'celery_diskutil']:
        proc_set[node] = complicated_status("/var/run/celery/%s.pid" % node, int)

    alerts = []
    for process, active in sorted(proc_set.items(), key=lambda s: s[0].lower()):
        print 'ok' if active else 'DOWN', process
        if not active:
            alerts.append(process)

    return alerts


def start_service(name):
    cmd = "service %s restart" % name
    print cmd
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        print 'Error: %s returned %d, %s, %s' % (cmd, proc.returncode, stdout, stderr)
    return proc.returncode == 0


def update_banner(alerts):
    new = False
    # update message banner
    message = Message.objects.filter(tags="service_status_alert")
    if len(alerts) > 0:
        msg = 'ALERT system services are down: %s. ' % ', '.join(alerts)
        msg += ' Please contact your system administrator for assistance.'
        if not message or message[0].body != msg:
            message.delete()
            Message.warn(msg, tags="service_status_alert")
            new = True
    else:
        message.delete()
    print '...updated message banner'
    return new


def send_alert(alerts):
    # send an email to IT contact
    msg = "The following system services are down:\n"
    for service in alerts:
        msg += service + '\n'
    notify_services_error('Torrent Server Alert', msg, msg.replace('\n','<br>'))
    print '...notified IT list'



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script can check and restart essential TS services")
    parser.add_argument('-n', '--notify', dest='notify', action='store_true', default=False, help='notify with banner and email alert')
    parser.add_argument('-s', '--start', dest='start', action='store_true', default=False, help='attempt to (re-)start all services')
    args = parser.parse_args()

    if args.start:
        if os.geteuid() != 0:
            sys.exit('Run this script with root permissions to start services')
        
        start_order = ['postgresql',
                       'apache2',
                       'gridengine-master', 'gridengine-exec',
                       'rabbitmq-server', 'celeryd', 'celerybeat',
                       'ionJobServer', 'ionCrawler', 'ionPlugin'
                       ]
        for name in start_order:
            start_service(name)

    # check status
    alerts = process_status()

    if args.notify:
        logger.info("check_system_services: %s" % alerts)
        
        try:
            new = update_banner(alerts)   
        except:
            logger.error('check_system_services: unable to update Message banner')
            print traceback.format_exc()
        else:
            # send an email to IT contact
            if alerts and new:
                try:
                    send_alert(alerts)
                except:
                    logger.error('check_system_services: unable to send email alert')
                    print traceback.format_exc()


