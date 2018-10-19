# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
import json
from datetime import timedelta

from celery.task import periodic_task
from django.core.cache import cache

from iondb.product_integration.models import ThermoFisherCloudAccount
from iondb.product_integration.utils import send_deep_laser_iot_request, get_deep_laser_instrument_status
from iondb.rundb.models import GlobalConfig


#@periodic_task(run_every=timedelta(minutes=1), queue="periodic")
def update_instrument_status_deep_laser():
    """ Sends the instrument status to deep laser periodically if it has changed. """

    # If deeplaser is disabled bail out
    gc = GlobalConfig.objects.get()
    if not gc.telemetry_enabled:
        return

    new_status = get_deep_laser_instrument_status()
    old_status = cache.get("instrument_status_deep_laser", "{}")

    if new_status != old_status:
        send_deep_laser_iot_request({
            "request": "updatedevicestatus",
            "status": json.loads(new_status)
        })

    cache.set("instrument_status_deep_laser", new_status, None)


def handle_deep_laser_device_request(key, parameters, principal_id=None):
    """ Handle a device request like disconnectDevice. Should raise an exception on error. """
    if key == "disconnectDevice":
        assert principal_id is not None
        for account in ThermoFisherCloudAccount.objects.all():
            account.delete()
    if key == "unlinkUser":
        for account in ThermoFisherCloudAccount.objects.filter(deeplaser_principleid=principal_id):
            account.delete()
        assert principal_id is not None
