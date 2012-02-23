# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
#!/usr/bin/env python
import sys
import os
sys.path.append('/opt/ion/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'iondb.settings'
from iondb.rundb import models

def get_selected_plugins():
    '''Build a list containing dictionaries of plugin information. 
    will only put the plugins in the list that are selected in the 
    interface'''
    try:
        pg = models.Plugin.objects.filter(selected=True)
    except:
        return ""
    ret = []
    if len(pg) > 0:
        ret = [{'name':p.name,
                'path':p.path,
                'version':p.version,
                'project':p.project,
                'sample':p.sample,
                'libraryName':p.libraryName,
                'chipType':p.chipType} for p in pg]
        return ret
    else:
        return ""
    
if __name__ == '__main__':
    plugins = get_selected_plugins()
    for plugin in plugins:
            print "Name: %s" % plugin['name']
            print "Path: %s" % plugin['path']
            print "Project: %s" % plugin['project']
            print "Sample: %s" % plugin['sample']
            print "Library: %s" % plugin['libraryName']
            print "ChipType: %s" % plugin['chipType']
            if plugin['project'] != None and plugin['project'] != "":
                print "Here we need to run a further test"
                for test in plugin['project'].split(','):
                    test = test.strip()
                    print "Is %s okay to run against?" % test
