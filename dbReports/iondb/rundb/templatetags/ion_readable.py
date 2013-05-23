# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
from django import template
import math


register = template.Library()
@register.filter(name='ion_readable')
def ion_readable(value):
    try:
        charlist = []
        charlist.append("")
        charlist.append(" K")
        charlist.append(" M")
        charlist.append(" G")

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
            if text[-1:] == '0':
                text = text.split('.')[0]
            textIntPart = text.split('.')[0]
            if len(textIntPart) > 2:
                text = textIntPart
            converted_text = str(text) + charlist[charindex]
        else:
            converted_text = str(value)

        return converted_text
    except:
        pass


@register.filter(name='float2int')
def float2int(value):
    try:
        return int(round(float(value)))
    except:
        pass


@register.filter(name='latexsafe')
def latexsafe(value):
    try:
        return_value = value.replace("_","\_")
        return_value = return_value.replace("%","\%")
        return return_value
    except:
        pass

@register.filter(name='bracewrap')
def bracewrap(value):
    return "{" + value + "}"


ion_readable.is_safe = True




ion_readable.is_safe = True
#register.filter(ion_readable)


