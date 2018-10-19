# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import json
import logging
from collections import OrderedDict
from django.core.urlresolvers import reverse
from iondb.rundb.plan.page_plan.abstract_step_data import AbstractStepData
from iondb.rundb.models import Plugin
from iondb.rundb.plan.page_plan.step_names import StepNames
from iondb.rundb.plan.page_plan.step_helper_types import StepHelperType

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
    IRU_QC_OPTIONS = 'iru_qc_option'


class PluginsStepData(AbstractStepData):

    def __init__(self, sh_type):
        super(PluginsStepData, self).__init__(sh_type)
        self.resourcePath = 'rundb/plan/page_plan/page_plan_plugins.html'
        self.prev_step_url = reverse("page_plan_kits")
        self.next_step_url = reverse("page_plan_output")
        if sh_type in StepHelperType.PLAN_BY_SAMPLE_TYPES:
            # Plan by Sample
            self.next_step_url = reverse("page_plan_by_sample_barcode")

        self.all_enabled_plugins = Plugin.objects.filter(selected=True, active=True).order_by('name', '-version')
        self.non_ir_plugins = []
        for p in self.all_enabled_plugins:
            info = p.info()
            if info:
                if PluginFieldNames.FEATURES in info:
                    # watch out: "Export" was changed to "export" recently!
                    # we now need to show all non-IRU export plugins on the Plugins chevron
                    # if (PluginFieldNames.EXPORT in (feature.lower() for feature in info[PluginFieldNames.FEATURES])):
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

        self.sh_type = sh_type

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
                PluginFieldNames.PLUGIN: plugin,
                PluginFieldNames.SELECTED: selected,
                PluginFieldNames.CONFIG: config
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
                    PluginFieldNames.PL_ID: plugin_id,
                    PluginFieldNames.NAME: values[PluginFieldNames.PLUGIN].name,
                    PluginFieldNames.VERSION: values[PluginFieldNames.PLUGIN].version,
                    PluginFieldNames.FEATURES: []}

                if values[PluginFieldNames.CONFIG]:
                    retval[values[PluginFieldNames.PLUGIN].name][PluginFieldNames.USER_INPUT] = json.loads(values[PluginFieldNames.CONFIG])
                else:
                    retval[values[PluginFieldNames.PLUGIN].name][PluginFieldNames.USER_INPUT] = ''
        return retval

    def isVariantCallerSelected(self):
        for plugin_id, values in self.savedObjects[PluginFieldNames.PLUGINS].items():
            if values[PluginFieldNames.PLUGIN].name == "variantCaller" and values[PluginFieldNames.SELECTED]:
                return True
        return False

    def isVariantCallerConfigured(self):
        for plugin_id, values in self.savedObjects[PluginFieldNames.PLUGINS].items():
            if values[PluginFieldNames.PLUGIN].name == "variantCaller" and values[PluginFieldNames.CONFIG]:
                return True
        return False

    def validateStep(self):
        """This method overrides the abstract class'es implementation and will validate each of the plugins individually."""

        # we don't need to perform this validation for templates
        if self.sh_type in StepHelperType.TEMPLATE_TYPES:
            return

        self.updateSavedObjectsFromSavedFields()
        # reset the validation errors
        self.validationErrors.clear()

        for plugin_id, values in self.savedObjects[PluginFieldNames.PLUGINS].items():
            plugin_model = Plugin.objects.get(id=plugin_id)
            if values[PluginFieldNames.SELECTED] and plugin_model.requires_configuration:
                configuration = dict() if values[PluginFieldNames.CONFIG] is None else json.loads(values[PluginFieldNames.CONFIG])
                plugin_validation_errors = Plugin.validate(plugin_id, configuration, 'Automatic')

                if plugin_validation_errors:
                    self.validationErrors[values[PluginFieldNames.PLUGIN].name] = plugin_validation_errors
