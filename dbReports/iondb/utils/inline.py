# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Inline.py

Utility functions for converting a file into a data URI.
"""

import base64 as b64
import mimetypes

DEFAULT_MIME_TYPE = "text/plain"


def urlinline(filename, mime=None):
    """
    Load the file at "filename" and convert it into a data URI with the
    given MIME type, or a guessed MIME type if no type is provided.

    Base-64 encodes the data.
    """
    infile = open(filename, 'rb')
    text = infile.read()
    infile.close()
    enc = b64.standard_b64encode(text)
    if mime is None:
        mime, _ = mimetypes.guess_type(filename)
        mime = mime or DEFAULT_MIME_TYPE
    ret = "data:%s;base64,%s" % (mime, enc)
    return ret
