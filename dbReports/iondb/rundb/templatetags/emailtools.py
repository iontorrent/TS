# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from os import path

from django import template
from django.conf import settings

from iondb.utils import inline

register = template.Library()

def check_token(token,rng):
    contents = token.split_contents()
    assert len(contents)
    name = contents[0]
    if len(contents)-1 not in rng:
        raise template.TemplateSyntaxError(
            "%r tag requires 1 or 2 arguments" % name)
    return contents[1:]

# Inline a file as a data URI

# tag format:
# {% inline_data <filname> <optional mimetype> %}
def do_inline_data(parser,token):
    contents = check_token(token,range(2,4))
    return DataURINode(*contents)

class DataURINode(template.Node):
    def __init__(self, filename, mimetype=None):
        super(DataURINode,self).__init__()
        self.filename = path.join(settings.MEDIA_ROOT,filename)
        self.mimetype = mimetype
    def render(self, context):
        return inline.urlinline(self.filename, self.mimetype)

register.tag('inline_data', do_inline_data)

# absolute-link using a base URI provided in settings.conf

def do_abs_url(parser,token):
    contents = check_token(token,range(1,2))
    return AbsURLNode(contents[0])

class AbsURLNode(template.Node):
    def __init__(self, suffix):
        super(AbsURLNode,self).__init__()
        self.suffix = suffix
    def render(self,context):
        return path.join(settings.ROOT_URL,self.suffix)

register.tag('abs_url', do_abs_url)

# absolute-link media

def do_abs_media(parser,token):
    contents = check_token(token,range(1,2))
    return AbsMediaNode(contents[0])

class AbsMediaNode(template.Node):
    def __init__(self, suffix):
        super(AbsMediaNode,self).__init__()
        self.suffix = suffix
    def render(self,context):
        return path.join(settings.ROOT_URL, "site_media", self.suffix)

register.tag('abs_media', do_abs_media)

# inline text

def do_inline_text(parser,token):
    contents = check_token(token,range(1,2))
    return InlineTextNode(contents[0])

class InlineTextNode(template.Node):
    def __init__(self, suffix):
        self.suffix = suffix
    def render(self, context):
        fname = path.join(settings.MEDIA_ROOT, self.suffix)
        infile = open(fname, 'rb')
        text = infile.read()
        infile.close()
        return text

register.tag('inline_text', do_inline_text)

