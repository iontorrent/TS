# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
from django import template
import math


register = template.Library()
@register.filter(name='ion_readable')
def ion_readable(value):
    charlist = []
    charlist.append("")
    charlist.append("K")
    charlist.append("M")
    charlist.append("G")

    charindex = 0
    val = float(value)
    while (val >= 1000):
        val = val / 1000
        charindex = charindex + 1

    converted_text = ""
    if (charindex > 0):
        val2 = math.floor(val*10)
        val2 = val2 / 10
        text = "%.1f" % val2
        converted_text = str(text) + charlist[charindex]
    else:
        converted_text = str(value)

    return converted_text

ion_readable.is_safe = True
#register.filter(ion_readable)


