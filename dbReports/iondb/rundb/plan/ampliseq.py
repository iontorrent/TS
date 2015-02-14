# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.models import Plugin

# This is the latest version of the TS for which AmpliSeq.com has needed 
# modification to accomodate.  For example, we could leave this string 
# "3.6" during the TS 4.0 release if the TS 4.0 release is compatible with
# the AmpliSeq output for the 3.6 TS in every way.
# Note: this is unlikely for most releases as a change to the Variant Caller,
# the Plan schema, or the BED publisher might necessitate changes here.
CURRENT_VERSION = "4.4"


def setup_vc_config_36(plan):
    vc_config = plan.pop("variant_caller", None)
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


def legacy_plan_handler(data):
    data["plan"] = dict(data)
    return data


def config_choice_handler_4_0(data, meta, config_choices):
    choice = meta.get("choice", None)
    keys = config_choices.keys()
    if len(keys) == 1 or choice not in keys:
        choice = sorted(keys)[0]
        meta["choice"] = choice
    plan = config_choices[choice]

    if "runType" in plan:
        if plan["runType"] == "AMPS_DNA":
            plan["runType"] = "AMPS"

    plan = setup_vc_config_36(plan)

    data["plan"] = plan
    data["configuration_choices"] = keys
    return data, meta


def plan_handler_4_4(data, meta):
    """
    current plan handler
    """
    config_choices = data["plan"]["4.4"]["configuration_choices"]
    return config_choice_handler_4_0(data, meta, config_choices)


def plan_handler_4_2(data, meta):
    config_choices = data["plan"]["4.2"]["configuration_choices"]
    return config_choice_handler_4_0(data, meta, config_choices)


def plan_handler_4_0(data, meta):
    config_choices = data["plan"]["4.0"]["configuration_choices"]
    return config_choice_handler_4_0(data, meta, config_choices)


def plan_handler_3_6(data, meta):
    plan = data["plan"]["3.6"]

    if "runType" in plan:
        if plan["runType"] == "AMPS_DNA":
            plan["runType"] = "AMPS"
    elif "pipeline" in data:
        if data["pipeline"] == "RNA":
            plan["runType"] = "AMPS_RNA"
        else:
            plan["runType"] = "AMPS"

    plan = setup_vc_config_36(plan)

    data["plan"] = plan
    data["configuration_choices"] = []
    return data, meta


version_plan_handlers = {
    "4.4": plan_handler_4_4,                        
    "4.2": plan_handler_4_2,
    "4.0": plan_handler_4_0,
    "3.6": plan_handler_3_6,
}


def handle_versioned_plans(data, meta=None):
    if meta is None:
        meta = {}
    # This is the very first iteration of AmpliSeq zip exports to be used
    # by the TS and it might not need to be supported at all
    if "plan" not in data:
        return "legacy", legacy_plan_handler(data), meta
    # The plan is empty or null
    elif not data['plan']:
        data["plan"] = {}
        return "unplanned", data, meta
    # This is the version we want to find, the version for *this* TS version
    # even if later versions are available in the JSON
    elif CURRENT_VERSION in data["plan"]:
        data, meta = version_plan_handlers[CURRENT_VERSION](data, meta)
        return CURRENT_VERSION, data, meta
    # If the current version isn't in there, it's because the zip is older
    # than the current version; however, it's possible that we know how
    # to handle archives from that older version for this TS version
    else:
        max_version = max(data["plan"].keys())
        if max_version in version_plan_handlers:
            data, meta = version_plan_handlers[max_version](data, meta)
            return max_version, data, meta
    return None, None, None
