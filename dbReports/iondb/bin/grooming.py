#!/usr/bin/env python

"""This will go through old log files and groom them."""

import os
os.chdir('/opt/ion')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iondb.settings")

import simplejson as json
from iondb.rundb.models import PluginResult

for result in PluginResult.objects.filter(plugin__name='RunTransfer'):
    # groom the database entry
    for plugin_result_job in result.plugin_result_jobs.all():
        if 'user_password' in plugin_result_job.config:
            del plugin_result_job.config['user_password']
            plugin_result_job.save()

    # groom the log file
    log_path = os.path.join(result.path(), "drmaa_stdout.txt")
    if os.path.exists(log_path):
        lines = [line for line in open(log_path) if 'user_password' not in line]
        open(log_path, 'w').writelines(lines)

    # groom the startplugin.json on the file system
    start_plugin_path = os.path.join(result.path(), "startplugin.json")
    if os.path.exists(start_plugin_path):
        start_plugin = dict()
        with open(start_plugin_path) as start_plugin_fp:
            start_plugin = json.load(start_plugin_fp)

        if 'pluginconfig' in start_plugin and 'user_password' in start_plugin['pluginconfig']:
            del start_plugin['pluginconfig']['user_password']
            with open(start_plugin_path, 'w') as start_plugin_fp:
                json.dump(start_plugin, start_plugin_fp, indent=4)

