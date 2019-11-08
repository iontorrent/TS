# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import re

_c = re.compile


class EntryParser(object):
    def __init__(self, regexp, coerce=None):
        self.coerce = coerce or extract_text
        self.regexp = regexp

    def __call__(self, line):
        match = self.regexp.match(line)
        if match is None:
            return None
        return self.coerce(match)


def extract_text(match):
    return match.groups(1)[0]


def intcoerce(match):
    return int(extract_text(match))


def floatcoerce(match):
    return float(extract_text(match))


def yncoerce(match):
    return extract_text(match).lower()[0] == "y"


def manyintcoerce(match, split=None):
    return map(int, extract_text(match).split(split))


simpleint = _c(r"^(\d+)$")
simplefloat = _c(r"^(\d+(?:\.(?:\d*))?)$")
y_slash_n = _c(r"^((?:yes)|(?:no))$", re.I)
text = _c(r"^(.*)$")
units = lambda unit: _c(r"^\s*(\d+)\s*%s*$" % unit)
space_separated_ints = _c(r"^((?:\d+\s+)*\d+)$")
n_ints = lambda n: _c(r"^((?:\d+\s+){%d}\d+)$" % (n - 1))
dot_separated = _c(r"^((?:\d+\.)+(?:\d+))$")
chip_name = _c(r"^(\d\d\d[A-Z])$")
kernel_build = _c(
    r"^([#]\d+\s+\w+\s+\w{3}\s+\w{3}\s+\d+\s+" "\d{2}[:]\d{2}[:]\d{2}\s+\w{3}\s+\d{4})$"
)
ref_electrode = _c(r"^$")

int_parse = EntryParser(simpleint, intcoerce)
float_parse = EntryParser(simplefloat, floatcoerce)
yn_parse = EntryParser(y_slash_n, yncoerce)
text_parse = EntryParser(text)
volts_parse = EntryParser(units("V"), intcoerce)
oversample_parse = EntryParser(units("x"), intcoerce)
space_separated_parse = EntryParser(space_separated_ints, manyintcoerce)
cal_range_parse = EntryParser(n_ints(3), manyintcoerce)
dac_parse = EntryParser(n_ints(8), manyintcoerce)
dot_separated_parse = EntryParser(dot_separated, lambda l: manyintcoerce(l, "."))
kernel_parse = EntryParser(kernel_build)
chip_parse = EntryParser(chip_name)
