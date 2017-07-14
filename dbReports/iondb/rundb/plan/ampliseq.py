# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.models import Plugin
import json
import os
import re
import sys
from distutils.version import StrictVersion
from ion import version as TS_version
from iondb.rundb.plan.ampliseq_validator import validate_reference, validate_bed_files, api
from iondb.rundb.plan.ampliseq_to_TS_plan_convertor import plan_json
"""
 This is the main class which handles the plan.json for the AmpliSeq Import

 This takes the  latest installed version of the TS for which AmpliSeq.com has needed
  modification to accomodate.
  Note: this is unlikely for most releases as a change to the Variant Caller,
  the Plan schema, or the BED publisher might necessitate changes here.
"""

class AmpliSeqPanelImport(object):
    def __init__(self, data=None, meta=None, arg_path=None):
        self.data = data
        self.meta = meta
        self.arg_path = arg_path
        self._config_choices = None
        self._plan = {}

    @property
    def config_choices(self):
        return self._config_choices

    @config_choices.setter
    def config_choices(self, value):
        self._config_choices = value

    @property
    def plan(self):
        return self._plan

    @plan.setter
    def plan(self, value):
        self._plan = value

    @staticmethod
    def get_errroMsg(errCode, value):
        errorDict = { "E001" : "ERROR: Invalid plugins found in the plan.json: {0}",
                      "E002" : "Unknown Error",
                      "E003":  "This panel is not backward compatible with Torrent Suite {0}. Please update Torrent suite or consult with TS/AmpliSeq.com Support Team."
                     }

        return errorDict[errCode].format(value)

    @staticmethod
    def get_TSversion():
        # return first 2 digit of TS version
        match = re.match(r'(([0-9]+\.[0-9]+)(\.[0-9]+)?)', str(TS_version))
        return (match.group(2))

    def get_choice_specific_planData(self):
        data = self.data
        meta = self.meta
        config_choices = self.config_choices
        choice = meta.get("choice", None)
        if choice == "p1":
            choice = "proton"
        config_instrument_data = config_choices.keys()

        if len(config_instrument_data) == 1 or choice not in config_instrument_data:
            choice = sorted(config_instrument_data)[0]
            # if user selects the wrong instrument type, TS should send validation error(TS-12754)
            # meta["choice"] = choice

        self.plan = config_choices[choice]
        plan = self.plan
        available_choice = []
        for available in config_instrument_data:
            available = str(available.upper())
            if available in ['520', '521', '530', '540']:
                available = "S5 Chip: " + available
            available_choice.append(available)
        plan["available_choice"] = available_choice
        if "runType" in plan:
            if plan["runType"] == "AMPS_DNA":
                plan["runType"] = "AMPS"

        return config_instrument_data

    def config_choice_handler(self, ampSeq_path=None):
        config_instrument_data = self.get_choice_specific_planData()
        data = self.data
        meta = self.meta
        self.plan = self.setup_VC_Plugin()

        if "plugins" in self.plan and not ampSeq_path:
            self.setup_other_plugin_config()

        data["plan"] = self.plan
        data["configuration_choices"] = config_instrument_data
        return data, meta

    # This is the main handle to process the plan.json plan data and config choices data
    def plan_handler(self):
        data = self.data
        meta = self.meta
        arg_path = self.arg_path

        config_choices = self.config_choices

        if not arg_path:
            data, meta = self.config_choice_handler(ampSeq_path=True)
        else:
            for key, value in config_choices.iteritems():
                if "plugins" in config_choices[key]:
                    allPluginPath = config_choices[key]['plugins']
                    for pluginName, paramFile in allPluginPath.iteritems():
                        specificPluginData = {}
                        if pluginName:
                            pluginName = pluginName.strip()
                            if pluginName == "variantCaller":
                                pluginName = "variant_caller"
                            if paramFile:
                                specificPluginData = json.load(open(os.path.join(arg_path, paramFile)))
                                config_choices[key][pluginName] = specificPluginData
                            else:
                                config_choices[key][pluginName] = {}
            data, meta = self.config_choice_handler()

        return data, meta

    # This handles the legacy TS version Plan.json - 3.6 and older
    def legacy_plan_handler(self):

        data = self.data
        meta = self.meta
        if self.plan_TS_version == "3.6":
            """This is the original version that was versioned in this"""
            plan = data["plan"]["3.6"]
            if "runType" in plan:
                if plan["runType"] == "AMPS_DNA":
                    plan["runType"] = "AMPS"
            elif "pipeline" in data:
                if data["pipeline"] == "RNA":
                    plan["runType"] = "AMPS_RNA"
                else:
                    plan["runType"] = "AMPS"
            self.plan = plan
            plan = self.setup_VC_Plugin()
            data["plan"] = plan
            data["configuration_choices"] = []
        else:
            """This is crudely supported.
            It will successfully import the BED file.
            """
            data["plan"] = dict(self.data)

    def setup_VC_Plugin(self):
        # This is required to support backward compatability
        plan = self.plan
        vc_config = plan.pop("variant_caller", None)
        plan["selectedPlugins"] = {}
        # If there's no VC config, we'll just skip the entire plugin configuration
        if vc_config is not None:
            name = "variantCaller"
            plugin = Plugin.objects.filter(name=name, active=True).order_by('-version')[0]

            # Add the VC config to the required frame JSON for the plan
            plan["selectedPlugins"] = {
                "variantCaller": {
                    "features": [],
                    "id": plugin.id,
                    "name": "variantCaller",
                    "userInput": vc_config,
                    "ampliSeqVariantCallerConfig": vc_config,
                    "version": plugin.version,
                }
            }
        return plan

    def setup_other_plugin_config(self):
        plan = self.plan
        plugin_config = plan.pop("plugins", None)
        inValidPlugins = self.validatePlugins(plugin_config)

        plugins = {}
        if not inValidPlugins:
            for pluginName, paramFile in plugin_config.iteritems():
                plugin_config = plan.pop(pluginName, None)
                # skip the variant caller since VC has already been configured by setup_vc_config
                # Proceed with other plugin configuration
                if pluginName and pluginName != 'variantCaller':
                    plugin = Plugin.objects.filter(name=pluginName, active=True).order_by('-version')[0]
                    # Add the other plugin config to the required frame JSON for the plan
                    pluginDict = {
                        "features": [],
                        "id": plugin.id,
                        "name": plugin.name,
                        "userInput": plugin_config,
                        "version": plugin.version
                    }
                    plugins[plugin.name] = pluginDict

            plan["selectedPlugins"].update(plugins)
        else:
            inValidPlugins_str = (', '.join(inValidPlugins))
            print (self.get_errroMsg("E001", inValidPlugins_str))
            sys.exit(1)

    def validatePlugins(self, plugin_config):
        inValidPlugins = []
        for pluginName, paramFile in plugin_config.iteritems():
            qs = Plugin.objects.filter(name=pluginName, active=True).order_by('id')
            if qs.count() == 0:
                inValidPlugins.append(pluginName)

        return inValidPlugins

    def handle_versioned_plans(self):
        """
         This validates and parses the plan.json JSON into an
         object meant to be read by the system in the current version.
         Also, handles the old and legacy Versions.
        """
        meta = self.meta
        data = self.data

        self.plan_TS_version = self.get_TSversion()

        if not meta:
            self.meta = {}
            meta = self.meta
        # This is the very first iteration of AmpliSeq zip exports to be used
        # by the TS and it might not need to be supported at all
        if "plan" not in data:
            self.legacy_plan_handler()
            return "legacy", self.data, meta
        elif not data['plan']: # handle if the plan is empty or null
            data["plan"] = {}
            return "unplanned", data, meta
        elif self.plan_TS_version in data["plan"]:
            # This is the version we want to find, the version for *this* TS version
            # even if later versions are available in the JSON
            configData = self.data["plan"][self.plan_TS_version]["configuration_choices"]

            # call the setter to set the configData
            self.config_choices = configData

            config_instrument_data = self.get_choice_specific_planData()
            self.plan_handler()
        else:
            """
            Choose appropriate plan parameters if there is no parameterized TS version available
            Example Scenario:
              - If current TS is 5.6, plan.json has sections for 5.8, 5.6, 5.0,
                    then plan definition for 5.6 should be used for template creation.
              - If current TS is 5.6, plan.json has sections for 5.8, 5.0,
                    then plan definition for 5.0 should be used for template creation.
              - if current TS is 5.6, plan.json has sections for 5.8,
                    then error message and no template creation.
            """
            planTS = self.plan_TS_version
            available_plan_version = [str(ver) for ver in data["plan"].keys()]
            available_plan_version.sort(key=StrictVersion)
            available_plan_version.reverse()

            isSubstitueVersionAvailable = False
            substitue_version = None
            if available_plan_version:
                for version in available_plan_version:
                    if float(version) < float(planTS):
                        isSubstitueVersionAvailable = True
                        substitue_version = version
                        break
            if not isSubstitueVersionAvailable:
                print (self.get_errroMsg("E003",TS_version))
                sys.exit(1)

            self.plan_TS_version = str(substitue_version)

            if "3.6" in self.plan_TS_version:
                self.legacy_plan_handler()
            else:
                configData = self.data["plan"][self.plan_TS_version]["configuration_choices"]
                self.config_choices = configData
                self.plan_handler()

        return self.data, self.meta

    def parse_ampliSeq_plan_json(self):
        # Get the choice speicific plan handle
        # Main method for all processing of plan.json
        self.handle_versioned_plans()

        isRef_genome_available = self.data.get('genome_reference', None)

        return self.meta, self.data, isRef_genome_available

def validate_ampliSeq_bundle(meta, args):
    plan_data = json.load(open(os.path.join(args.path, "plan.json")))

    ampliSeq = AmpliSeqPanelImport(data=plan_data, meta=meta, arg_path=args.path)

    # parse the the ampliseq bundle plan.json and get the chip/choice specific design data and meta data
    meta, design, reference =  ampliSeq.parse_ampliSeq_plan_json()
    meta['design'] = design

    # wait for the reference install:
    # finish_me subtask will be called after reference install to restart publisher upload
    isRefInstallInProgress  = validate_reference(meta, args, reference)

    if not isRefInstallInProgress:
        if not meta.get("reference"):
            meta['reference'] = design.get('genome','')
        if 'design_name' in design:
            meta['description'] = design['design_name']
        
        validate_bed_files(meta, args)

    return isRefInstallInProgress, meta


def convert_AS_to_TS_plan_and_post(
                                   meta,
                                   args,
                                   target_regions_bed_path,
                                   hotspots_bed_path,
                                   sse_bed_path):

    plan_prototype, alignmentargs_override =  plan_json(meta, args.upload_id, target_regions_bed_path, hotspots_bed_path, sse_bed_path)

    return post_TS_plan(plan_prototype, alignmentargs_override, args)


def post_TS_plan(plan_prototype, alignmentargs_override, args):
    isUploadFailed = False
    errMsg = None
    try:
        success, response, content = api.post("plannedexperiment", **plan_prototype)
        if not success:
            api.patch("contentupload", args.upload_id, status="Error: unable to create TS Plan")
            err_content = json.loads(content)
            error_message_array = []
            if 'error' in err_content:
                error_json = json.loads(str(err_content['error'][3:-2]))
                for k in error_json:
                    for j in range(len(error_json[k])):
                        err_message = str(error_json[k][j])
                        err_message = err_message.replace('&gt;', '>')
                        error_message_array.append(err_message)
            error_messages = ','.join(error_message_array)
            raise Exception(error_messages)
        if alignmentargs_override:
            content_dict = json.loads(content)
            api.patch("plannedexperiment",
                      content_dict["id"],
                      alignmentargs=alignmentargs_override,
                      thumbnailalignmentargs=alignmentargs_override)

    except Exception as err:
        isUploadFailed = True
        errMsg = err

    return success, isUploadFailed, errMsg
