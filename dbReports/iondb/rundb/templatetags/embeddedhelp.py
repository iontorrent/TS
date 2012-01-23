# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from django import template
from django.template import loader
from django.conf import settings

import emailtools

register = template.Library()

def do_tooltip(parser,token):
    contents = emailtools.check_token(token,range(1,2))
    return TooltipNode(contents[0])

class TooltipNode(template.Node):
    def __init__(self, tip):
        self.tip = tip
    def render(self,context):
        tmpl = loader.get_template("rundb/tooltip.html")
        ctx = template.Context({"tip":self.tip})
        return tmpl.render(ctx)

register.tag("tooltip", do_tooltip)    

@register.inclusion_tag("rundb/tooltip_summary.html")
def tooltip_summary():
    return {}
