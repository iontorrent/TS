# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
from collections import Iterable

from django import template
from django.template import loader
from django.conf import settings
import os

import emailtools
from django.template.base import TemplateSyntaxError, Node, Variable

register = template.Library()


def do_near_range(parser, token):
    contents = emailtools.check_token(token, range(1, 2))
    return NearRangeNode(contents[0])


class NearRangeNode(template.Node):
    def __init__(self, pagname):
        super(NearRangeNode, self).__init__()
        self.pagname = pagname

    def render(self, context):
        p = context[self.pagname].paginator
        ndx = context[self.pagname].number
        npages = p.num_pages
        width = 6
        halfw = width / 2
        if npages < width:
            width = npages
        if ndx <= halfw:
            first = 1
            last = first + width
        elif ndx >= (npages - halfw):
            last = npages + 1
            first = last - width
        else:
            first = ndx - halfw
            last = ndx + halfw
        context["page_number_range"] = range(first, last)
        return ""


register.tag("near_range", do_near_range)


def do_icon(parser, token):
    icons = [ic.strip('"') for ic in emailtools.check_token(token, range(1, 3))]
    urlnodes = parser.parse(("endicon",))
    parser.delete_first_token()
    nodeargs = [urlnodes] + icons
    return IconNode(*nodeargs)


class IconNode(template.Node):
    def __init__(self, urlnodes, icon1, icon2=None):
        self.urlnodes = urlnodes
        self.icon1 = icon1
        self.icon2 = icon2

    def render(self, context):
        tmpl = loader.get_template("rundb/icon.html")
        url = self.urlnodes.render(context).strip()
        ctx = template.Context({"url": url, "icon1": self.icon1, "icon2": self.icon2})
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


def basename(filepath):
    return os.path.basename(filepath)


register.tag("icon", do_icon)
register.filter("blankIfNone", blankIfNone)
register.filter("boxChecked", boxChecked)
register.filter("fileName", fileName)
register.filter("basename", basename)

# raw tag parser function copyright 2009, EveryBlock
# This code is released under the GPL.
@register.tag
def raw(parser, token):
    # Whatever is between {% raw %} and {% endraw %} will be preserved as
    # raw, unrendered template code.
    text = []
    parse_until = "endraw"
    tag_mapping = {
        template.TOKEN_TEXT: ("", ""),
        template.TOKEN_VAR: ("{{", "}}"),
        template.TOKEN_BLOCK: ("{%", "%}"),
        template.TOKEN_COMMENT: ("{#", "#}"),
    }
    # By the time this template tag is called, the template system has already
    # lexed the template into tokens. Here, we loop over the tokens until
    # {% endraw %} and parse them to TextNodes. We have to add the start and
    # end bits (e.g. "{{" for variables) because those have already been
    # stripped off in a previous part of the template-parsing process.
    while parser.tokens:
        token = parser.next_token()
        if token.token_type == template.TOKEN_BLOCK and token.contents == parse_until:
            return template.TextNode(u"".join(text))
        start, end = tag_mapping[token.token_type]
        text.append(u"%s%s%s" % (start, token.contents, end))
    parser.unclosed_block_tag(parse_until)


@register.filter(name="bracket")
def bracket(value, arg):
    try:
        return value[arg]
    except Exception:
        return None


# settings value
@register.tag
def value_from_settings(parser, token):
    bits = token.split_contents()
    if len(bits) < 2:
        raise TemplateSyntaxError(
            "'%s' takes at least one argument (settings constant to retrieve)" % bits[0]
        )
    settingsvar = bits[1]
    settingsvar = settingsvar[1:-1] if settingsvar[0] == '"' else settingsvar
    asvar = None
    bits = bits[2:]
    if len(bits) >= 2 and bits[-2] == "as":
        asvar = bits[-1]
        bits = bits[:-2]
    if len(bits):
        raise TemplateSyntaxError(
            "'value_from_settings' didn't recognise the arguments '%s'"
            % ", ".join(bits)
        )
    return ValueFromSettings(settingsvar, asvar)


class ValueFromSettings(Node):
    def __init__(self, settingsvar, asvar):
        self.arg = Variable(settingsvar)
        self.asvar = asvar

    def render(self, context):
        ret_val = getattr(settings, str(self.arg))
        if self.asvar:
            context[self.asvar] = ret_val
            return ""
        else:
            return ret_val


@register.filter
def makeColumns(mylist, ncol):
    # reshape a list to display n columns
    try:
        l = len(mylist)
        ncol = int(ncol)
    except Exception:
        return mylist
    nrows = 1 + int(l / ncol) if (l % ncol > 0) else int(l / ncol)
    return [mylist[nrows * i : nrows * (i + 1)] for i in range(ncol) if i < l]


# Custom filter to check if a template exists
@register.filter
def template_exists(template_name):
    try:
        template.loader.get_template(template_name)
        return True
    except template.TemplateDoesNotExist:
        return False


@register.simple_tag
def model_name(value):
    """
    Django template filter which returns the verbose name of a model.
    """
    if hasattr(value, "model"):
        value = value.model

    return value._meta.verbose_name.title()


@register.simple_tag
def model_verbose_name(value):
    """
    Django template filter which returns the verbose name of a model.
    """
    if hasattr(value, "model"):
        value = value.model

    return value._meta.verbose_name.title()


@register.simple_tag
def model_verbose_name_plural(value):
    """
    Django template filter which returns the plural verbose name of a model.
    """
    if hasattr(value, "model"):
        value = value.model

    return value._meta.verbose_name_plural.title()


@register.simple_tag
def field_verbose_name(value, field):
    """
    Django template filter which returns the verbose name of an object's,
    model's or related manager's field.
    """
    if hasattr(value, "model"):
        value = value.model

    return value._meta.get_field(field).verbose_name.title()


@register.filter
def verbose_name(obj):
    return obj._meta.verbose_name


@register.filter
def verbose_name_plural(obj):
    return obj._meta.verbose_name_plural


@register.filter
def is_string(val):
    return isinstance(val, basestring)


@register.filter
def is_iterable(val):
    return isinstance(val, Iterable)
