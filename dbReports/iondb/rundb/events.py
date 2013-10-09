# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import logging
import httplib2
import json
import pytz
import datetime

from django import http
from django.db.models import signals
from tastypie.serializers import Serializer

from iondb.rundb import models
from iondb.rundb import api

logger = logging.getLogger(__name__)
serializer = Serializer()


def build_event_response(sender, when, data):
    """Create a container object with some meta data similar to a tastypie response"""
    event = {
        'meta': {
            'event_domain': sender,
            'event_name': when,
            'timestamp': datetime.datetime.now(tz=pytz.utc),
            'attributes': [],
        },
        'object': data,
    }
    return serializer.to_json(event)


def post_event(body, consumer):
    try:
        h = httplib2.Http()
        response, content = h.request(consumer, method="POST", body=body, headers={"Content-type": "application/json"})
        logger.debug("Posting Event %s: %s %s" % (consumer, response, content))
    except Exception as err:
        logger.exception("Post event error: %s" % err)


def model_event_post(domain, name, sender_model, instance, producer, consumer):
    data = producer(sender_model, instance)
    result = build_event_response(domain, name, data)
    return post_event(result, consumer)


def listen_created(domain, name, sender_model, producer, consumer):
    logger.info("Registering %s %s %s" % (sender_model, producer, consumer))
    def handler(sender, instance, created, **kwargs):
        logger.warning("testing chandler %s" % sender)
        if created:
            logger.warning("running chandler %s" % sender)
            model_event_post(domain, name, sender, instance, producer, consumer)
    signals.post_save.connect(handler, sender=sender_model)
    return signals.post_save, handler


def listen_saved(domain, name, sender_model, producer, consumer):
    logger.info("Registering %s %s %s" % (sender_model, producer, consumer))
    def handler(sender, instance, created, **kwargs):
        logger.warning("calling shandler %s" % sender)
        model_event_post(domain, name, sender, instance, producer, consumer)
    signals.post_save.connect(handler, sender=sender_model)
    return signals.post_save, handler


def listen_deleted(domain, name, sender_model, producer, consumer):
    logger.info("Registering %s %s %s" % (sender_model, producer, consumer))
    def handler(sender, instance, **kwargs):
        logger.warning("calling dhandler %s" % sender)
        model_event_post(domain, name, sender, instance, producer, consumer)
    signals.pre_delete.connect(handler, sender=sender_model)
    return signals.pre_delete, handler


def logging_omni_handler(*args, **kwargs):
    "This is a dummy signal handler that just logs whatever comes it's way."
    logger.info("Dummy args %s kwargs %s" % (str(args), str(kwargs)))
    return "Output args %s kwargs %s" % (str(args), str(kwargs))


def api_serializer(model_resource):
    resource = model_resource()
    def producer(sender, instance):
        logger.debug("Running %s producer" % resource)
        bundle = resource.build_bundle(obj=instance)
        dry = resource.full_dehydrate(bundle)
        logger.debug("Producing %s: %s" % (type(resource), str(dry)))
        return dry
    return producer



event_when = {
    'save': listen_saved,
    'create': listen_created,
    'delete': listen_deleted
}


event_senders = {
    'Experiment': (models.Experiment, api_serializer(api.ExperimentResource)), 
    'Result': (models.Results, api_serializer(api.ResultsResource)),
    'Plan': (models.PlannedExperiment, api_serializer(api.PlannedExperimentResource)),
    'Rig': (models.Rig, api_serializer(api.RigResource)),
    'Plugin': (models.Plugin, api_serializer(api.PluginResource)),
    'PluginResult': (models.PluginResult, api_serializer(api.PluginResultResource)),
}


event_handlers = []


def register_events(event_consumers):
    "Read the settings' event config and attach to the relevant django signals"
    global event_handlers
    event_handlers = []
    for (sender, when), consumers in event_consumers.items():
        try:
            sender_model, producer = event_senders[sender]
            listen = event_when[when]
            for consumer in consumers:
                signal_info = listen(sender, when, sender_model, producer, consumer)
                event_handlers.append(signal_info)
                logger.info("Successfully registered " + sender + ' ' + when)
        except Exception as err:
            logger.exception("Register error %s" % err)


def remove_events(events=None):
    "This disconnects all of the event API handlers from their related Django signals"
    events = events or event_handlers
    logger.debug(events)
    for signal_info in events:
        try:
            logger.debug(signal_info)
            signal, handler = signal_info
            signal.disconnect(handler)
            event_handlers.remove(signal_info)
        except Exception as err:
            logger.exception("Remove error")


def demo_consumer(request, name=""):
    """A django view whose sole function is to log what is POSTed to it"""
    try:
        data = json.loads(request.body)
        text = json.dumps(data, sort_keys=True, indent=4)
        logger.info("Demo Consumer %s:\n %s" % (name, text))
        return http.HttpResponse(text)
    except Exception as err:
        logger.exception("Demo Consumer")
        return http.HttpResponseServerError()