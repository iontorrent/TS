# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from django import template
from django.template import loader
from django.conf import settings
import os

import emailtools

register = template.Library()

def do_near_range(parser,token):
    contents = emailtools.check_token(token, range(1,2))
    return NearRangeNode(contents[0])

class NearRangeNode(template.Node):
    def __init__(self, pagname):
        super(NearRangeNode,self).__init__()
        self.pagname = pagname
    def render(self, context):
        p = context[self.pagname].paginator
        ndx = context[self.pagname].number
        npages = p.num_pages
        width = 6
        halfw = width/2
        if npages < width:
            width = npages
        if ndx <= halfw:
            first = 1
            last = first + width
        elif ndx >= (npages-halfw):
            last = npages+1
            first = last - width
        else:
            first = ndx - halfw
            last = ndx + halfw
        context["page_number_range"] = range(first,last)
        return ""
        
register.tag("near_range", do_near_range)

def do_icon(parser,token):
    icons = [ic.strip('"') for ic in emailtools.check_token(token, range(1,3))]
    urlnodes = parser.parse(('endicon',))
    parser.delete_first_token()
    nodeargs = [urlnodes] + icons
    return IconNode(*nodeargs)

class IconNode(template.Node):
    def __init__(self,urlnodes,icon1,icon2=None):
        self.urlnodes = urlnodes
        self.icon1 = icon1
        self.icon2 = icon2
    def render(self, context):
        tmpl = loader.get_template("rundb/icon.html")
        url = self.urlnodes.render(context).strip()
        ctx = template.Context({"url":url, "icon1":self.icon1,
                                "icon2":self.icon2})
        return tmpl.render(ctx)


def blankIfNone(value):
    if value:
        return value
    else:
        return ""

def boxChecked(value):
    if value == False:
        return ""
    else:
        return "CHECKED"

def fileName(value):
    return os.path.splitext(os.path.basename(value))[0]

register.tag("icon", do_icon)
register.filter("blankIfNone", blankIfNone)
register.filter("boxChecked", boxChecked)
register.filter("fileName", fileName)

