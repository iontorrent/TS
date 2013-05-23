# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

from iondb.rundb.models import Plugin

# This is the latest version of the TS for which AmpliSeq.com has needed 
# modification to accomodate.  For example, we could leave this string 
# "3.6" during the TS 4.0 release if the TS 4.0 release is compatible with
# the AmpliSeq output for the 3.6 TS in every way.
# Note: this is unlikely for most releases as a change to the Variant Caller,
# the Plan schema, or the BED publisher might necessitate changes here.
CURRENT_VERSION = "3.6"


def legacy_plan_handler(data):
	data["plan"] = dict(data)
	return data

# Current plan handler
def plan_handler_3_6(data):
	plan = data["plan"]["3.6"]
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
	data["plan"] = plan
	return data



version_plan_handlers = {
	"3.6": plan_handler_3_6,
}


def handle_versioned_plans(data):
	# This is the very first iteration of AmpliSeq zip exports to be used
	# by the TS and it might not need to be supported at all
	if "plan" not in data:
		return "legacy", legacy_plan_handler(data)
	# This is the version we want to find, the version for *this* TS version
	# even if later versions are available in the JSON
	elif CURRENT_VERSION in data["plan"]:
		return CURRENT_VERSION, version_plan_handlers[CURRENT_VERSION](data)
	# If the current version isn't in there, it's because the zip is older
	# than the current version; however, it's possible that we know how
	# to handle archives from that older version for this TS version
	else:
		max_version = max(data["plan"].keys())
		if max_version in version_plan_handlers:
			return max_version, version_plan_handlers[max_version](data)
	return None, None
