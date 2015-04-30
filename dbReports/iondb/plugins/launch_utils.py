# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

import json
from iondb.rundb.models import Plugin, PluginResult
from ion.plugin.constants import RunLevel

import logging
logger = logging.getLogger(__name__)

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
    for p in pg:
        params = {
            'name':p.name,
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
                accountId = params['userInput']['accountId']
            except:
                accountId = ''
            if 'userconfigs' in p.config:
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
        active_plugins_deps[name] = settings.get('depends', [])
    
    plugin_names = plugins.keys()
    dep = {}
    for name in plugin_names:
        if name not in active_plugins_deps:
            continue # plugin no longer installed
        # add deeper dependencies 
        for dependency in active_plugins_deps[name]:
            if dependency not in active_plugins_deps:
                logger.error("Plugin %s requested dependency on %s, which isn't installed", name, dependency)
                #del plugins[name]
                #break
            if dependency not in plugin_names:
                plugin_names.append(dependency)
        else:
            # for...else runs if loop doesn't break, so only add to dep list if all dependencies found.
            dep[name] = active_plugins_deps[name]
    
    sorted_names = toposort(dep)
    
    # get existing plugin results
    plugin_results = PluginResult.objects.filter(result__pk = pk)
    
    # update plugins to add dependencies
    satisfied_dependencies = {}
    for name in sorted_names:
        if name not in plugins.keys():
            pr = plugin_results.filter(plugin__name=name).filter(state__in=['Completed','Started','Queued']).order_by('-pk')
            if pr:
                satisfied_dependencies[name] = {
                    'pluginresult': pr[0].pk,
                    'version': pr[0].plugin.version,
                    'pluginresult_path': pr[0].path()
                }
            # else:
            #     # add dependency plugin to be launched
            #     try:
            #         p = active_plugins.get(name=name)
            #         plugins.update(get_plugins_dict([p]))
            #     except Plugin.DoesNotExist:
            #         logger.exception("Plugin requested dependency on %s, which isn't installed", name)

            
    return plugins, sorted_names, satisfied_dependencies

def get_plugins_to_run(plugins, result_pk, runlevel):
    # updates plugins with dependencies
    # gets plugins to run for this runlevel
    plugins, sorted_names, satisfied_dependencies = depsolve(plugins, result_pk)
    plugins_to_run = []
    for name in sorted_names:
        if name in plugins:
            plugin_runlevels = plugins[name].get('runlevel') or [RunLevel.DEFAULT]
            if runlevel in plugin_runlevels:
                plugins_to_run.append(name)
    
    return plugins, plugins_to_run, satisfied_dependencies
    
    
def add_hold_jid(plugin, plugins_dict, runlevel):
    # add hold_jid values for any dependencies
    if 'hold_jid' not in plugin.keys():
        plugin['hold_jid'] = []
    
    for name in plugin.get('depends',[]):
        if name in plugins_dict and plugins_dict[name].get('jid'):            
            plugin['hold_jid'].append(plugins_dict[name]['jid'])

    # multilevel plugins: add holds for themselves
    multilevel_jid = plugin.get('jid','')
    if multilevel_jid and multilevel_jid not in plugin['hold_jid']:
        plugin['hold_jid'].append(plugin['jid'])
    
    # post runlevel gathers block runs
    if runlevel == RunLevel.POST:
        plugin['hold_jid'] += plugin.get('block_jid',[])
        
    # last runlevel holds for all previously launched plugins
    if runlevel == RunLevel.LAST:
        plugin['hold_jid'] = [p['jid'] for p in plugins_dict.values() if p.get('jid')]
    
    return plugin
