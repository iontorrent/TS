# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved

import json
import logging
import time
import uuid

import requests

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
