# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
from django import template
import math


register = template.Library()
@register.filter(name='ion_readable')
def ion_readable(value):
    try:
        charlist = ["", " K", " M", " G", " T"]
        charindex = 0
        val = float(value)
        while (round(val) >= 1000):
            val = val / 1000
            charindex = charindex + 1

        converted_text = ""
        if (charindex > 0):
            for n in [0, 1, 2]:
                text = str(round(val, n))
                if text.split('.')[1] == "0":
                    text = text.split('.')[0]
                if len(text.replace('.','')) > 2:
                    break
            converted_text = str(text) + charlist[charindex]
        else:
            converted_text = str(value)

        return converted_text
    except:
        pass


@register.filter(name="cleanName")
def clean_name(name):
    if name.startswith("R_"):
        return name[27:]
    else:
        return name


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


@register.filter(name='chunks')
def chunks(iterable, chunk_size):
    """chunks takes an iterable and yields a series of iterables of chunk_size
    or fewer elements.  This is best used for breaking up a long list, into rows
    of chunk_size elements, for example.
    """
    if not hasattr(iterable, '__iter__'):
        # can't use "return" and "yield" in the same function
        yield iterable
    else:
        i = 0
        chunk = []
        for item in iterable:
            chunk.append(item)
            i += 1
            if not i % chunk_size:
                yield chunk
                chunk = []
        if chunk:
            # some items will remain which haven't been yielded yet,
            # unless len(iterable) is divisible by chunk_size
            yield chunk



ion_readable.is_safe = True




ion_readable.is_safe = True
#register.filter(ion_readable)


