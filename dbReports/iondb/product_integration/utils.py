# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
import json
import logging
import time
import uuid

import requests
from django.core.serializers.json import DjangoJSONEncoder
from ion import version as TS_version
from ion.utils.makeSSA import get_servicetag

from iondb.rundb.models import GlobalConfig

logger = logging.getLogger(__name__)


def send_deep_laser_iot_request(request_dict):
    """ Sends a iot request to the deep laser client via http. Returns the request id."""
    assert type(request_dict) == dict

    request_id = request_dict.get("id")
    if not request_id:
        request_id = str(uuid.uuid4())
        request_dict["id"] = request_id

    logger.debug(
        "Posting iot request to DL request_id=%s\n%s\n",
        request_id,
        json.dumps(request_dict, cls=DjangoJSONEncoder, sort_keys=True, indent=2),
    )

    deep_laser_response = requests.post(
        "http://localhost:9000/v1/iotrequest", data=json.dumps(request_dict, cls=DjangoJSONEncoder)
    )
    logger.debug(
        "sync response to above iot request from DL request_id=%s\nstatus_code: %s\n%s\n",
        request_id,
        deep_laser_response.status_code,
        json.dumps(deep_laser_response.json(), cls=DjangoJSONEncoder, sort_keys=True, indent=2),
    )

    assert deep_laser_response.status_code == 200

    return request_id


def send_deep_laser_metrics(event_type, event_data, schema=1):
    if schema == 1:
        payload = {
            "request": "senddata",
            "appName": "Torrent Suite",
            "appVersion": TS_version,
            "datatype": "metric",
            "eventId": event_type,
            "parameters": event_data,
        }
        payload["parameters"]["serverSerial"] = get_servicetag()
        payload["parameters"]["siteName"] = GlobalConfig.get().site_name
        return send_deep_laser_iot_request(payload)
    elif schema == 2:
        raise DeprecationWarning
        # payload = {
        #     "request": "senddata",
        #     "application": "Torrent Suite",
        #     "applicationVersion": TS_version,
        #     "timestamp": int(time.time()),
        #     "datatype": "metric",
        #     "eventId": event_type,
        #     "eventParameters": event_data,
        # }
        # payload["eventParameters"]["serverSerial"] = get_servicetag()
        # payload["eventParameters"]["siteName"] = GlobalConfig.get().site_name
        # return send_deep_laser_iot_request(payload)
    else:
        raise ValueError("send_deep_laser_metrics schema must be 1 or 2")


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
    logger.debug(
        "Posting device response to DL \n%s\n",
        json.dumps(request_dict, sort_keys=True, indent=2),
    )

    deep_laser_response = requests.post(
        "http://localhost:9000/v1/deviceresponse", data=json.dumps(request_dict, cls=DjangoJSONEncoder)
    )
    logger.debug(
        "sync response to above device request from DL \nstatus_code: %s\n%s\n",
        deep_laser_response.status_code,
        json.dumps(deep_laser_response.json(), sort_keys=True, indent=2, cls=DjangoJSONEncoder),
    )

    assert deep_laser_response.status_code == 200
