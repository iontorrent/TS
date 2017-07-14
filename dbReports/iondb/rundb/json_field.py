# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
JSONField automatically serializes most Python terms to JSON data.
Creates a TEXT field with a default value of "{}".  See test_json.py for
more information.

 from django.db import models
 from django_extensions.db.fields import json

 class LOL(models.Model):
     extra = json.JSONField()
"""

import datetime
import json
from decimal import Decimal
from django.db import models
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder
from math import isnan

from json import encoder
encoder.FLOAT_REPR = lambda x: format(x, '.15g')

def groom_for_json(value):
    """Helper method to remove all invalid values from a dictionary before encoding to json"""

    def groom_list_for_json(my_list):
        for index in range(len(my_list)):
            my_list[index] = groom_for_json(my_list[index])
        return my_list

    def groom_dict_for_json(my_dictionary):
        for key in my_dictionary.keys():
            my_dictionary[key] = groom_for_json(my_dictionary[key])
        return my_dictionary

    if isinstance(value, dict):
        return groom_dict_for_json(value)
    elif isinstance(value, list):
        return groom_list_for_json(value)
    elif isinstance(value, float) and isnan(value):
        return "NaN"
    else:
        return value


class JSONEncoder(DjangoJSONEncoder):

    """ Override datetime.datetime representation. Defer to Django for other encodings """

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            assert settings.TIME_ZONE == 'UTC'
            return obj.strftime('%Y-%m-%dT%H:%M:%SZ')
        return super(JSONEncoder, self).default(obj)


def dumps(value):
    assert isinstance(value, dict)
    return json.dumps(value, cls=JSONEncoder, separators=(',', ':'))


def loads(txt):
    try:
        value = json.loads(
            txt,
            parse_float=Decimal,
            encoding=settings.DEFAULT_CHARSET
        )
        assert isinstance(value, dict)
    except (TypeError, ValueError):
        value = {}
    return value


class JSONDict(dict):

    """
    Hack so repr() called by dumpdata will output JSON instead of
    Python formatted data.  This way fixtures will work!
    """

    def __repr__(self):
        return dumps(self)


class JSONField(models.TextField):

    """JSONField is a generic textfield that neatly serializes/unserializes
    JSON objects seamlessly.  Main thingy must be a dict object."""

    # Used so to_python() is called
    __metaclass__ = models.SubfieldBase

    def __init__(self, *args, **kwargs):
        if 'default' not in kwargs:
            kwargs['default'] = '{}'
        models.TextField.__init__(self, *args, **kwargs)

    def to_python(self, value):
        """Convert our string value to JSON after we load it from the DB"""
        if not value:
            return {}
        elif isinstance(value, basestring):
            res = loads(value)
            assert isinstance(res, dict)
            return JSONDict(**res)
        else:
            return value

    def get_db_prep_save(self, value, connection=None):
        """Convert our JSON object to a string before we save"""
        if not value:
            return super(JSONField, self).get_db_prep_save("", connection=connection)
        else:
            value = groom_for_json(value)
            return super(JSONField, self).get_db_prep_save(dumps(value), connection=connection)

    def south_field_triple(self):
        "Returns a suitable description of this field for South."
        # We'll just introspect the _actual_ field.
        from south.modelsinspector import introspector
        field_class = "django.db.models.fields.TextField"
        args, kwargs = introspector(self)
        # That's our definition!
        return (field_class, args, kwargs)

