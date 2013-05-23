# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
import logging
logger = logging.getLogger(__name__)
from django.db.models import Q
from tastypie.bundle import Bundle
from iondb.rundb import models
from django.conf import settings


def base_context_processor(request):
    """This is a hook which adds the returned dictionary to the context for
    every view in the project.  It is used to allow the site name to be
    added to the base_template in every view.
    Namespace any items added here with 'base_' in order to avoid conflict.
    """
    site_name = models.GlobalConfig.objects.all().order_by('id')[0].site_name
    messages = models.Message.objects.filter(route="").filter(
        Q(status="unread", expires="read") | ~Q(expires="read"))
    from iondb.rundb.api import MessageResource
    resource = MessageResource()
    msg_list = [resource.full_dehydrate(Bundle(message)) for message in messages]
    serialized_messages = resource.serialize(None, msg_list, "application/json")
    base_js_extra = settings.JS_EXTRA
    if msg_list:
        logger.debug("Found %d global messages" % len(msg_list))
        
    logger.debug("Global messages are %s" % serialized_messages)
    
    if request.user:
        user_messages = models.Message.objects.filter(route=request.user).filter(Q(status="unread", expires="read") | ~Q(expires="read"))
        user_msglist = [resource.full_dehydrate(Bundle(message)) for message in user_messages]
        user_serialized_messages = resource.serialize(None, user_msglist, "application/json")
        logger.debug("User messages are %s" % user_serialized_messages)
    return {"base_site_name": site_name, "global_messages": serialized_messages,
            "user_messages":user_serialized_messages, "base_js_extra" : base_js_extra}


def message_binding_processor(request):
    """This is called for every request but only performs work for
    those which have bindings for specific message routes.
    """
    if hasattr(request, "message_bindings"):
        messages = models.Message.objects.bound(*request.message_bindings) \
            .filter(Q(status="unread", expires="read") | ~Q(expires="read"))
        bound_messages = list(messages)
        messages.update(status="read")
        if bound_messages:
            logger.debug("Found %d bound messages" % len(bound_messages))
        return {"bound_messages": bound_messages}
    else:
        logger.debug("No messages found")
        return {}


def bind_messages(*bindings):
    """This returns a function which takes a view and decorates it with
    a wrapper function which adds the binding information passed to
    this function to the request object for that view so that the
    message_binding_processor can access the specific bindings for the
    view that goes through it.  This feels like a serious hack...
    """
    def bind_view(view_func):
        def bound_view(request, *args, **kwargs):
            request.message_bindings = bindings
            return view_func(request, *args, **kwargs)
        logger.debug("Binding %s to %s" % (str(bindings),
                                           str(view_func.__name__)))
        return bound_view
    return bind_view
