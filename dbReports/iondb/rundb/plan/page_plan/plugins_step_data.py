# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import json
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import Plugin
from iondb.rundb.plan.page_plan.step_names import StepNames
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import logging
logger = logging.getLogger(__name__)

class PluginFieldNames():

    PL_ID = 'id'
    NAME = 'name'
    VERSION = 'version'
    FEATURES = 'features'
    EXPORT = 'export'
    PLUGIN = 'plugin'
    PLUGINS = 'plugins'
    PLUGIN_IDS = 'pluginIds'
    PLUGIN_CONFIG = 'plugin_config_%s'
    PLUGIN_ID_LIST = 'pluginIdList'
    SELECTED = 'selected'
    CONFIG = 'config'
    USER_INPUT = 'userInput'
    ACCOUNT_ID = 'accountId'
    ACCOUNT_NAME = 'accountName'

class PluginsStepData(AbstractStepData):

    def __init__(self):
        super(PluginsStepData, self).__init__()
        self.resourcePath = 'rundb/plan/page_plan/page_plan_plugins.html'
        self.all_enabled_plugins = Plugin.objects.filter(selected=True, active=True).order_by('name', '-version')
        self.non_ir_plugins = []
        for p in self.all_enabled_plugins:
            info = p.info()
            if info:
                if PluginFieldNames.FEATURES in info:
                    #watch out: "Export" was changed to "export" recently!
                    #we now need to show all non-IRU export plugins on the Plugins chevron
                    ##if (PluginFieldNames.EXPORT in (feature.lower() for feature in info[PluginFieldNames.FEATURES])):
                    if "ionreporter" in p.name.lower():
                        pass
                    else:
                        self.non_ir_plugins.append(p)
                else:
                    self.non_ir_plugins.append(p)
        
        
        self.prepopulatedFields[PluginFieldNames.PLUGINS] = self.non_ir_plugins
        for plugin in self.prepopulatedFields[PluginFieldNames.PLUGINS]:
            if plugin.isPlanConfig:
                self.savedFields[PluginFieldNames.PLUGIN_CONFIG % plugin.id] = None

        self.savedFields[PluginFieldNames.PLUGIN_IDS] = None
        self.savedListFieldNames.append(PluginFieldNames.PLUGINS)
        self.savedObjects[PluginFieldNames.PLUGINS] = OrderedDict()
        self.savedObjects[PluginFieldNames.PLUGIN_ID_LIST] = []
        self.updateSavedObjectsFromSavedFields()

    def getStepName(self):
        return StepNames.PLUGINS

    def updateSavedObjectsFromSavedFields(self):
        self.savedObjects[PluginFieldNames.PLUGINS].clear()
        self.savedObjects[PluginFieldNames.PLUGIN_ID_LIST] = []
        
        if self.savedFields[PluginFieldNames.PLUGIN_IDS]:
            self.savedObjects[PluginFieldNames.PLUGIN_ID_LIST] = self.savedFields[PluginFieldNames.PLUGIN_IDS].split(', ')
        
        for plugin in self.prepopulatedFields[PluginFieldNames.PLUGINS]:
            selected = False
            if self.savedObjects[PluginFieldNames.PLUGIN_ID_LIST] and (str(plugin.id) in self.savedObjects[PluginFieldNames.PLUGIN_ID_LIST] or
                                                plugin.id in self.savedObjects[PluginFieldNames.PLUGIN_ID_LIST]):
                selected = True
            
            config = None
            if plugin.isPlanConfig and self.savedFields[PluginFieldNames.PLUGIN_CONFIG % plugin.id]:
                config = json.dumps(json.loads(self.savedFields[PluginFieldNames.PLUGIN_CONFIG % plugin.id]))
                
            self.savedObjects[PluginFieldNames.PLUGINS][plugin.id] = {
                                        PluginFieldNames.PLUGIN    : plugin, 
                                        PluginFieldNames.SELECTED  : selected, 
                                        PluginFieldNames.CONFIG    : config
                                        }
    
    def updateFromStep(self, updated_step):
        pass
    
    def getSelectedPluginsValue(self):
        retval = {}
        
        if not self.savedObjects[PluginFieldNames.PLUGIN_ID_LIST]:
            return retval
        
        for plugin_id, values in self.savedObjects[PluginFieldNames.PLUGINS].items():
            if values[PluginFieldNames.SELECTED]:
                retval[values[PluginFieldNames.PLUGIN].name] = {
                                        PluginFieldNames.PL_ID     : plugin_id,
                                        PluginFieldNames.NAME      : values[PluginFieldNames.PLUGIN].name,
                                        PluginFieldNames.VERSION   : values[PluginFieldNames.PLUGIN].version,
                                        PluginFieldNames.FEATURES  : []}

                if values[PluginFieldNames.CONFIG]:
                    retval[values[PluginFieldNames.PLUGIN].name][PluginFieldNames.USER_INPUT] = json.loads(values[PluginFieldNames.CONFIG])
                else:
                    retval[values[PluginFieldNames.PLUGIN].name][PluginFieldNames.USER_INPUT] = ''
        return retval
