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
    gconfig = models.GlobalConfig.get()
    messages = models.Message.objects.filter(route="").filter(
        Q(status="unread", expires="read") | ~Q(expires="read")
    )
    from iondb.rundb.api import MessageResource

    resource = MessageResource()
    msg_list = [resource.full_dehydrate(Bundle(message)) for message in messages]
    serialized_messages = resource.serialize(None, msg_list, "application/json")
    base_js_extra = settings.JS_EXTRA

    if request.user:
        users = [request.user.username]
        if request.user.is_staff:
            users.append(models.Message.USER_STAFF)
        user_messages = models.Message.objects.filter(route__in=users).filter(
            Q(status="unread", expires="read") | ~Q(expires="read")
        )
        user_msglist = [
            resource.full_dehydrate(Bundle(message)) for message in user_messages
        ]
        user_serialized_messages = resource.serialize(
            None, user_msglist, "application/json"
        )

    unread_news = 0
    if settings.FEATURE_FLAGS.NEWS:
        if request.user.is_authenticated() and gconfig.check_news_posts:
            unread_news = models.NewsPost.objects.filter(
                updated__gte=request.user.userprofile.last_read_news_post
            ).count()

    return {
        "base_site_name": gconfig.site_name,
        "global_messages": serialized_messages,
        "user_messages": user_serialized_messages,
        "base_js_extra": base_js_extra,
        "unread_news": unread_news,
        "DEBUG": settings.DEBUG,
        "version": settings.VERSION,
        "featureflags": settings.FEATURE_FLAGS,
    }


def message_binding_processor(request):
    """This is called for every request but only performs work for
    those which have bindings for specific message routes.
    """
    if hasattr(request, "message_bindings"):
        messages = models.Message.objects.bound(*request.message_bindings).filter(
            Q(status="unread", expires="read") | ~Q(expires="read")
        )
        bound_messages = list(messages)
        messages.update(status="read")
        return {"bound_messages": bound_messages}
    else:
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

        logger.debug("Binding %s to %s" % (str(bindings), str(view_func.__name__)))
        return bound_view

    return bind_view


def add_help_urls_processor(request):
    """This function adds helps urls to the template context. Url patterns are limited to exact matches and
    globs after a slash. This allows us to only make as many dict lookups as there are
    slashes in the url. Much faster than actually globing each pattern or regex."""

    help_urls = []

    # Break the url up by /s
    request_url_fragments = request.get_full_path().split("/")[1:-1]

    # Check if there are any help urls for the specific url
    # Example Url Pattern: /plan/plan_templates/
    help_urls.extend(
        settings.HELP_URL_MAP.get("/" + "/".join(request_url_fragments) + "/", [])
    )

    # Check if there are any help urls for parent urls including this one
    # Example Url Pattern: /plan/*
    for i in range(1, len(request_url_fragments) + 1):
        parent_url_fragments = request_url_fragments[0:i]
        help_urls.extend(
            settings.HELP_URL_MAP.get("/" + "/".join(parent_url_fragments) + "/*", [])
        )

    return {"help_urls": help_urls}
