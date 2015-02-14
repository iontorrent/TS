# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved

import collections
import logging
logger = logging.getLogger(__name__)

def convert(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data