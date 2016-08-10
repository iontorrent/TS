# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from django.core.cache import cache
import json


def apply_prepopulated_values_to_step_helper(request, step_helper):
    # Prefill Sample Names
    if "Save_plan" in step_helper.steps:
        samples_table = json.loads(step_helper.steps["Save_plan"].savedFields["samplesTable"])
        for i, sample in enumerate(samples_table):
            if "sampleName" not in sample or sample["sampleName"] == "":
                sample["sampleName"] = "Sample %i" % (i+1)
        step_helper.steps["Save_plan"].savedFields["samplesTable"] = json.dumps(samples_table)
        step_helper.steps["Save_plan"].updateSavedObjectsFromSavedFields()

    # Torrent Hub Sessions
    if "prepopulated-planning-session" in request.GET:
        key_name = "prepopulated-planning-session-" + request.GET["prepopulated-planning-session"]
        cached_values = cache.get(key_name)
        if cached_values:
            # Loop through the planning data and seed values on the step_helper
            # Ex {"Save_plan.planName": "Orange Seq Run 1"}
            for step_name, step_fields in cached_values["step_helper"].items():
                for field_name, value in step_fields.items():
                    step_helper.steps[step_name].savedFields[field_name] = value

            # Call some util methods on some steps to parse savedFields data
            for step in step_helper.steps.values():
                try:
                    step.updateSavedObjectsFromSavedFields()
                except Exception as ex:
                    pass

            # Add a redirect url to the session
            request.session["post_planning_redirect_url"] = cached_values["redirect_url"]

            cache.delete(key_name)
