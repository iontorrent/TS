# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
from datetime import timedelta
from utils import send_deep_laser_metrics
from celery.task import periodic_task
from metrics import get_server_metrics, get_report_metrics, get_plugin_metrics


@periodic_task(run_every=timedelta(hours=1), queue="periodic")
def send_deep_laser_server_metrics():
    """ Sends health metrics periodically """
    send_deep_laser_metrics("SERVER_METRICS", get_server_metrics(), schema=1)


@periodic_task(run_every=timedelta(hours=24), queue="periodic")
def send_deep_laser_report_metrics():
    """ Sends report metrics periodically """
    send_deep_laser_metrics("NEW_REPORT", get_report_metrics(hours=36), schema=1)


@periodic_task(run_every=timedelta(hours=24), queue="periodic")
def send_deep_laser_plugin_metrics():
    """ Sends plugin metrics periodically """
    send_deep_laser_metrics("NEW_PLUGIN", get_plugin_metrics(hours=36), schema=1)


def handle_deep_laser_device_request(key, parameters, principal_id=None):
    """ Handle a device request like disconnectDevice. Should raise an exception on error. """
    pass
