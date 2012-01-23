# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

from django import template
from django.template import loader
from django.conf import settings

register = template.Library()

@register.inclusion_tag('rundb/mobile/table_row.html')
def table_row(label, value, link=None):
    link_external = link and not link.startswith('#')
    return {'label': label, 'value': value, 'link' : link, 'link_external' : link_external}

@register.inclusion_tag('rundb/mobile/grid_row.html')
def grid_row(label, values):
    return {'label': label, 'values': values}

@register.inclusion_tag('rundb/mobile/report_img.html')
def report_img(helper, filename, alt="", num_in_row=1):
    width_class = "full_width" if num_in_row == 1 else "half_width"
    return {'has_image': helper.has_file(filename), 'image_link': helper.link_for_file(filename),
            'alt': alt, 'width_class': width_class}

@register.filter
def has_file(report_helper, filename):
    return report_helper.has_file(filename)

@register.filter
def file_link(report_helper, filename):
    return report_helper.link_for_file(filename)

@register.filter
def format(value, fmt):
    """
    Alters default filter "stringformat" to not add the % at the front,
    so the variable can be placed anywhere in the string.
    """
    try:
        if value:
            return fmt % value
        else:
            return ''
    except (ValueError, TypeError):
        return ''

@register.filter
def space_to_underscore(value):
    return value.replace(' ', '_')
