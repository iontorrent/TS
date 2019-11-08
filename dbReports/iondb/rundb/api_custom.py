# Copyright (C) 2018 Ion Torrent Systems, Inc. All Rights Reserved
import csv
import os
import json
import StringIO

from django.conf import settings
from django.utils.cache import patch_cache_control, patch_vary_headers
from django.views.decorators.csrf import csrf_exempt

from tastypie import fields, http
from tastypie.constants import ALL, ALL_WITH_RELATIONS
from tastypie.exceptions import BadRequest
from tastypie.resources import (
    ModelResource as _ModelResource,
    Resource,
    ModelDeclarativeMetaclass,
    sanitize,
)
from tastypie.serializers import Serializer
from tastypie.utils.formatting import format_datetime

# auth
from tastypie.authorization import DjangoAuthorization
from iondb.rundb.authn import IonAuthentication

import traceback
import logging

logger = logging.getLogger(__name__)


def field_dict(field_list):
    """will build a dict with to tell TastyPie to allow filtering by everything"""
    field_dict = {}
    field_dict = field_dict.fromkeys(field_list, ALL_WITH_RELATIONS)
    return field_dict


def JSONconvert(self, value):
    """If a CharFiled is really a Python dict, keep it as a dict """
    if value is None:
        return None
    # if it is a dict don't make it a string
    if isinstance(value, dict):
        return value

    return unicode(value)


class SDKValidationError(BadRequest):
    """
    Raised when a validation error is discovered for the request.
    Introduced to avoid django.core.exceptions.ValidationError since it is very particular
    about the structure of the nested object graph it is provided.
    """

    def errors(self):
        return self.args[0] if getattr(self, "args") else ""


class CustomSerializer(Serializer):
    formats = settings.TASTYPIE_DEFAULT_FORMATS
    content_types = {
        "json": "application/json",
        "jsonp": "text/javascript",
        "xml": "application/xml",
        "yaml": "text/yaml",
        "html": "text/html",
        "plist": "application/x-plist",
        "csv": "text/csv",
    }

    def format_datetime(self, data):
        """
        A hook to control how datetimes are formatted.

        DO NOT RETURN NAIVE datetime objects!!
        tastypie.utils.timezone.make_naive() is stripping our datetime object's tzinfo
        because django.utils.timezone.get_default_timezone() is returning a timezone of None.
        Django is using None because DJANGO_SETTINGS_MODULE defines TIME_ZONE=None.
        The timezone is set to None to default to use the OS timezone. This custom serializer
        to format datetime objects overriddes this undesired behavior.
        """
        # data = make_naive(data)
        if self.datetime_formatting == "rfc-2822":
            return format_datetime(data)
        data = data.replace(microsecond=int(round(data.microsecond / 1000)))
        return data.isoformat()

    def to_csv(self, data, options=None):
        options = options or {}
        data = self.to_simple(data, options)
        raw_data = StringIO.StringIO()
        first = True
        writer = None
        for item in data.get("objects", {}):
            if first:
                writer = csv.DictWriter(
                    raw_data, fieldnames=list(item.keys()), extrasaction="ignore"
                )
                # writer.writeheader() # python 2.7+
                writer.writerow(dict(zip(writer.fieldnames, writer.fieldnames)))
            writer.writerow(item)
        raw_data.seek(0)
        return raw_data.getvalue()

    def from_csv(self, content):
        raw_data = StringIO.StringIO(content)
        with csv.Sniffer() as sniffer:
            has_header = sniffer.has_header(raw_data.read(1024))
        raw_data.seek(0)
        data = []
        with csv.DictReader(raw_data) as reader:
            if has_header:
                fieldnames = reader.next()
            for item in reader:
                data.append(item)
        return data


class MyModelDeclarativeMetaclass(ModelDeclarativeMetaclass):
    def __new__(cls, name, bases, attrs):
        meta = attrs.get("Meta")

        if meta and not hasattr(meta, "serializer"):
            setattr(meta, "serializer", CustomSerializer())
        if meta and not hasattr(meta, "default_format"):
            setattr(meta, "default_format", "application/json")
        if meta and not hasattr(meta, "authentication"):
            setattr(meta, "authentication", IonAuthentication())
        if meta and not hasattr(meta, "authorization"):
            setattr(meta, "authorization", DjangoAuthorization())

        new_class = super(MyModelDeclarativeMetaclass, cls).__new__(
            cls, name, bases, attrs
        )

        return new_class


class ModelResource(_ModelResource):
    __metaclass__ = MyModelDeclarativeMetaclass

    def apply_sorting(self, obj_list, options=None):
        obj_list = super(ModelResource, self).apply_sorting(obj_list, options=options)
        # If the model has no default ordering AND the query has not been ordered by the above call
        # then sort the list by -pk so that pagination works as expected.
        if (
            len(obj_list.model._meta.ordering) == 0
            and len(obj_list.query.order_by) == 0
        ):
            return obj_list.order_by("-" + obj_list.model._meta.pk.name)
        else:
            return obj_list

    def wrap_view(self, view):
        """
        Overridding tastypie.resources.Resource to introduce a new except clause to preserve
        existing v1 response structure depsite misusage of tastypie's SDK.

            except SDKValidationError as e:
                data = {"error": e.args[0] if getattr(e, 'args') else ''}
                return self.error_response(request, data, response_class=http.HttpBadRequest)
        --

        Wraps methods so they can be called in a more functional way as well
        as handling exceptions better.

        Note that if ``BadRequest`` or an exception with a ``response`` attr
        are seen, there is special handling to either present a message back
        to the user or return the response traveling with the exception.
        """

        @csrf_exempt
        def wrapper(request, *args, **kwargs):
            try:
                callback = getattr(self, view)
                response = callback(request, *args, **kwargs)

                # Our response can vary based on a number of factors, use
                # the cache class to determine what we should ``Vary`` on so
                # caches won't return the wrong (cached) version.
                varies = getattr(self._meta.cache, "varies", [])

                if varies:
                    patch_vary_headers(response, varies)

                if self._meta.cache.cacheable(request, response):
                    if self._meta.cache.cache_control():
                        # If the request is cacheable and we have a
                        # ``Cache-Control`` available then patch the header.
                        patch_cache_control(
                            response, **self._meta.cache.cache_control()
                        )

                if request.is_ajax() and not response.has_header("Cache-Control"):
                    # IE excessively caches XMLHttpRequests, so we're disabling
                    # the browser cache here.
                    # See http://www.enhanceie.com/ie/bugs.asp for details.
                    patch_cache_control(response, no_cache=True)

                return response
            except SDKValidationError as e:
                data = {"error": e.errors()}
                return self.error_response(
                    request, data, response_class=http.HttpBadRequest
                )
            except (BadRequest, fields.ApiFieldError) as e:
                data = {"error": sanitize(e.args[0]) if getattr(e, "args") else ""}
                return self.error_response(
                    request, data, response_class=http.HttpBadRequest
                )
            except SDKValidationError as e:
                data = {"error": sanitize(e.messages)}
                return self.error_response(
                    request, data, response_class=http.HttpBadRequest
                )
            except Exception as e:
                if hasattr(e, "response"):
                    return e.response

                # A real, non-expected exception.
                # Handle the case where the full traceback is more helpful
                # than the serialized error.
                if settings.DEBUG and getattr(settings, "TASTYPIE_FULL_DEBUG", False):
                    raise

                # Re-raise the error to get a proper traceback when the error
                # happend during a test case
                if request.META.get("SERVER_NAME") == "testserver":
                    raise

                # Rather than re-raising, we're going to things similar to
                # what Django does. The difference is returning a serialized
                # error message.
                return self._handle_500(request, e)

        return wrapper


def getAPIexamples(url=None):
    api_examples_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "api_examples.json"
    )
    try:
        with open(api_examples_file) as f:
            data = json.load(f)
            if url:
                data = data[url]
    except Exception:
        logger.error(traceback.format_exc())
        data = "No example response available"

    return data
