# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import json
from iondb.rundb.models import Plugin, PluginResult

def find_IRU_account(pluginconfig, accountId):
    if accountId:
        for username,configs in pluginconfig.get('userconfigs',{}).items():
            for config in configs:
                if accountId == config.get('id'):
                    return config
    return {}
    

def get_plugins_dict(pg, selectedPlugins=''):
    """
    Build a list containing dictionaries of plugin information.
    Add userInput, if any, from selectedPlugins json
    """
    ret = {}
    if len(pg) > 0:
        
        for p in pg:
            params = {'name':p.name,
                  'path':p.path,
                  'version':p.version,
                  'id':p.id,
                  'pluginconfig': dict(p.config),
                  } 
            
            if selectedPlugins:
                params['userInput'] = selectedPlugins.get(p.name, {}).get('userInput','')
                # with TS4.0 need to get IRU config based on selected account
                if "IonReporterUploader" in p.name:
                    try:
                        accountId = params['userInput'].get('accountId')
                    except:
                        accountId = ""
                    if accountId:
                        params['pluginconfig'] = find_IRU_account(p.config, accountId)
                  
            for key in p.pluginsettings.keys():
                params[key] = p.pluginsettings[key]
            
            ret[p.name] = params
    return ret


def toposort(graph):
    '''
    Does topological sort of graph dict ( {'one': ['two','three'], 'two':['three'], etc} )
        and returns ordered list of keys
    Cyclic dependencies are ignored.
    '''
    def get(n,d):
        if n not in done:
            done.append(n)
            for m in d[n]:
                get(m, d)
            ordered_list.append(n)

    # check for missing keys
    for value in set(sum(graph.values(), [])):
        if value not in graph.keys():
            graph[value] = []
    
    # sort
    done = []
    ordered_list = []
    for key in graph.keys():
        get(key,graph)
        
    return ordered_list


def depsolve(plugins, pk):
    '''
    Adds any plugins that are dependent on
    Returns updated list sorted in topological order
    '''    
    # create dependency dict for all active plugins
    active_plugins = Plugin.objects.filter(selected=True,active=True).exclude(path='')
    active_plugins_deps = {}
    for name,settings in active_plugins.values_list('name','pluginsettings'):
        try:
            settings = json.loads(settings)
        except:
            settings = {}
        active_plugins_deps[name] = settings.get('depends') if settings.get('depends') else []
    
    plugin_names = plugins.keys()
    dep = {}
    for name in plugin_names:        
        if name not in active_plugins_deps:
            continue # plugin no longer installed
        dep[name] = active_plugins_deps[name]
        # add deeper dependencies 
        for dependency in active_plugins_deps[name]:
            if dependency not in plugin_names:
                plugin_names.append(dependency)
    
    sorted_names = toposort(dep)
    
    # get existing plugin results with state to check whether dependency needs to be launched    
    plugin_results = dict( [(r[0], r[1]) for r in PluginResult.objects.filter(result__pk = pk).order_by('pk').values_list('plugin__name', 'state')] )
    
    # update plugins to add dependencies
    for name in sorted_names:
        if name not in plugins.keys():
            # add dependency plugin unless it's already running or done
            if name in plugin_results and plugin_results[name] in ['Completed','Started','Queued']:
                continue
            else:
                p = active_plugins.get(name=name)
                plugins.update(get_plugins_dict([p]))
            
    return plugins, sorted_names

    
def add_hold_jid(plugin, plugins_to_run, runlevel):
    # add hold_jid values for any dependencies
    if 'hold_jid' not in plugin.keys():
        plugin['hold_jid'] = []
    
    for name in plugin.get('depends',[]):
        if name in plugins_to_run.keys():            
            plugin['hold_jid'].append(plugins_to_run[name].get('jid',''))

    # multilevel plugins: add holds for themselves
    multilevel_jid = plugin.get('jid','')
    if multilevel_jid and multilevel_jid not in plugin['hold_jid']:
        plugin['hold_jid'].append(plugin['jid'])
    
    # post runlevel gathers all block runs
    if runlevel == 'post':
        plugin['hold_jid'] += plugin.get('block_jid',[])
    
    return plugin
