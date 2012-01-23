#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from os import path
import re

COMMENT_LINE_RE = re.compile(r'^\s*#.*$',re.I)
TOOLTIP_LINE_RE = re.compile(r'^\s*"(?P<key>\w+)"\s*"(?P<title>[^"]+)"'
                             r'\s*(?P<text>.*)\s*$',re.I)

TOOLTIP_FILE = path.join(path.abspath(path.dirname(__file__)), "tooltips.txt")
#print TOOLTIP_FILE

def _format(title,text):
    return ({"title":title,"text":text},False)

_TIPS = {}

def reload_tooltips(fname=TOOLTIP_FILE):
    infile = open(fname)
    ret = []
    for line in infile:
        line = line.strip()
        if not line:
            continue
        elif COMMENT_LINE_RE.match(line):
            continue
        match = TOOLTIP_LINE_RE.match(line)
        if match:
            d = match.groupdict()
            ret.append((d['key'],_format(d['title'],d['text'])))
        else:
            raise ValueError("Invalid line: %s" % line)
    infile.close()
    for k,v in ret:
        _TIPS[k] = v
        

def default(s):
    return "No tooltip for key '%s'." % s

def tip(s):
    if s in _TIPS:
        ret = _TIPS[s]
        if isinstance(ret,(list,tuple)) and len(ret) == 2:
            ret,encoded = ret
        else:
            encoded = False
        if callable(ret):
            out = ret(s)
        else:
            out = ret
        return out,encoded
    else:
        return default(s),False

reload_tooltips()
