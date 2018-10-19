# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

import datetime
import json
import logging
import time
import uuid

import pytz
import requests

from iondb.rundb.home.runs import get_runs_list

logger = logging.getLogger(__name__)


def send_deep_laser_iot_request(request_dict):
    """ Sends a iot request to the deep laser client via http. Returns the request id."""
    assert type(request_dict) == dict

    request_id = request_dict.get("id")
    if not request_id:
        request_id = str(uuid.uuid4())
        request_dict["id"] = request_id

    logger.debug("Posting iot request to DL request_id=%s\n%s\n",
                 request_id,
                 json.dumps(request_dict, sort_keys=True, indent=2))

    deep_laser_response = requests.post("http://localhost:9000/v1/iotrequest", data=json.dumps(request_dict))
    logger.debug("sync response to above iot request from DL request_id=%s\nstatus_code: %s\n%s\n",
                 request_id,
                 deep_laser_response.status_code,
                 json.dumps(deep_laser_response.json(), sort_keys=True, indent=2)
                 )

    assert deep_laser_response.status_code == 200

    return request_id


def get_deep_laser_iot_response(request_id, timeout=30):
    """ Polls the models for a specific request id and then returns the model. """
    # Avoid circular import
    from iondb.product_integration.models import DeepLaserResponse
    assert type(request_id) == str
    attempts = 0
    while attempts < timeout * 2:
        attempts += 1
        response = DeepLaserResponse.objects.filter(request_id=request_id).first()
        if response:
            return response
        time.sleep(0.5)
    raise ValueError("No iot response from deep laser for request id: %s" % request_id)


def send_deep_laser_device_response(request_dict):
    """ Sends an http response to the deep laser client to ack a device request. """
    assert type(request_dict) == dict
    logger.debug("Posting device response to DL \n%s\n",
                 json.dumps(request_dict, sort_keys=True, indent=2))

    deep_laser_response = requests.post("http://localhost:9000/v1/deviceresponse", data=json.dumps(request_dict))
    logger.debug("sync response to above device request from DL \nstatus_code: %s\n%s\n",
                 deep_laser_response.status_code,
                 json.dumps(deep_laser_response.json(), sort_keys=True, indent=2)
                 )

    assert deep_laser_response.status_code == 200


def get_deep_laser_instrument_status():
    """ Generates the instrument status object to send to DL on a regular basis """
    # Reuse the function used by the dashboard
    raw_runs = get_runs_list(datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=7))
    # Pull out a list of fields we want to send to TFC
    # Mostly to reduce the size of the document
    run_fields = {"runType", "applicationCategoryDisplayedName", "last_updated", "uid", "error_string",
                  "progress_string", "state", "date", "stage", "name", "instrumentName"}
    filtered_runs = []
    for raw_run in raw_runs:
        filtered_run = {}
        for key in run_fields:
            filtered_run[key] = raw_run.get(key)
            if type(filtered_run[key]) == datetime.datetime:
                filtered_run[key] = filtered_run[key].strftime('%Y-%m-%dT%H:%M:%S')
        filtered_runs.append(filtered_run)

    # The object that is the status document
    instrument_status = {
        "status": "Idle",
        "schemaVersion": 1,
        "runs": filtered_runs
    }
    # Compute the global instrument status

    # AWS shadow document size limit is 8kB
    # Try to reduce the list of runs to fit
    while len(json.dumps(instrument_status)) >= 8000 and len(instrument_status["runs"]) > 0:
        logger.debug("DL instrument status is too large, dropping last run row")
        instrument_status["runs"].pop()

    return json.dumps(instrument_status)
