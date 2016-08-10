# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import re
import urllib

INTERNAL_ESCAPES = (
    ('/', '&'),
    ('\\', '$'),
    ('.', '#'),
)


def escape_name(name):
    ret = urllib.quote(name)
    for src, tgt in INTERNAL_ESCAPES:
        ret = ret.replace(src, tgt)
    return ret


def unescape_name(name):
    for src, tgt in INTERNAL_ESCAPES:
        name = name.replace(tgt, src)
    return urllib.unquote(name)


class NameField(object):
    MATCH = r"[^.]"
    ENCODE = True

    def __init__(self, name=None):
        self.name = name
        self.rank = None

    def get_field(self, s):
        return s

    def set_field(self, s):
        return s


class RawNameField(NameField):
    MATCH = r"."
    ENCODE = False


class IntegerField(NameField):
    MATCH = r"\d"
    ENCODE = True

    def get_field(self, s):
        return int(s)


class BooleanField(NameField):
    MATCH = r"[TrueFals]"
    ENCODE = True

    def test(self, s):
        return s == 'True'

    def get_field(self, s):
        return self.test(s)

    def set_field(self, s):
        return str(self.test(s))


class ConstantField(NameField):
    MATCH = r"[^.]"
    ENCODE = True

    def check_string(self, s):
        if s != self.name:
            raise ValueError, ("'%s' was not expected in constant field '%s'"
                               % (s, self.name))

    def get_field(self, s):
        self.check_string
        return self.name

    def set_field(self, s):
        return self.name


class ChoiceField(NameField):
    MATCH = r"[^.]"
    ENCODE = True

    def __init__(self, name=None, choices=None):
        NameField.__init__(self, name)
        if choices is None:
            raise ValueError, "No choices specified for ChoiceField."
        self.choices = set()
        for c in choices:
            self.choices.add(str(c))

    def check_str(self, s):
        if not s in self.choices:
            raise ValueError, "Unknown choice: '%s'." % s

    def get_field(self, s):
        self.check_str(s)
        return s

    def set_field(self, s):
        self.check_str(s)
        return s


class FileExtension(object):

    def __init__(self, name, gzipped=False):
        self.name = name
        self.gzipped = gzipped
        self.match = [''.join(["[%s]" % c for c in self.name])]
        self.to_sub = [self.name]
        if self.gzipped:
            self.match.append("gz")
            self.to_sub.append("gz")

    def __str__(self):
        if self.gzipped:
            base = "%s.gz"
        else:
            base = "%s"
        return base % self.name


class FieldMaster(object):

    def __init__(self):
        self.fields = []
        self.finalized = False
        self.file_ext = None
        self.regex = None
        self.to_sub = None
        self.name2field = None

    def set_ext(self, ext):
        if self.finalized:
            raise ValueError, (
                "Cannot set suffix for finalized %s"
                % self.__class__.__name__)
        if self.file_ext is not None:
            raise ValueError, "Attempted to re-set FieldMaster file extension."
        self.file_ext = ext

    def add_field(self, field):
        if self.finalized:
            raise ValueError, (
                "Cannot add field to finalized %s"
                % self.__class__.__name__)
        field.rank = len(self.fields)
        self.fields.append(field)

    def finalize(self):
        self.fields.sort(lambda a, b: cmp(a.name, b.name))
        regex = []
        to_sub = []
        self.name2field = {}
        for field in self.fields:
            regex.append(r"(?P<%s>%s*)" % (field.name, field.MATCH))
            to_sub.append("%%(%s)s" % field.name)
            self.name2field[field.name] = field
        if self.file_ext is not None:
            regex.extend(self.file_ext.match)
            to_sub.extend(self.file_ext.to_sub)
        tomatch = "^%s$" % '.'.join(regex)
        to_sub = ".".join(to_sub)
        self.regex = re.compile(tomatch)
        self.to_sub = to_sub
        self.finalized = True

    def parse_string(self, s):
        match = self.regex.match(s)
        if match is None:
            return None
        d = match.groupdict()
        for k, v in d.iteritems():
            f = self.name2field[k]
            if f.ENCODE:
                v = unescape_name(v)
            d[k] = self.name2field[k].get_field(v)
        return d

    def to_string(self, fnameobj):
        subfields = {}
        for field in self.fields:
            raw_val = field.set_field(str(getattr(fnameobj, field.name)))
            if field.ENCODE:
                val = escape_name(raw_val)
            else:
                val = raw_val
            subfields[field.name] = val
        return self.to_sub % subfields

    def init_fnameobj(self, fnameobj):
        for f in self.fields:
            setattr(fnameobj, f.name, None)


class FileNameBase(type):

    def make_prop(self, field):
        pass

    def __new__(cls, name, bases, clsdict):
        retdict = {}
        fm = FieldMaster()
        for k, v in clsdict.iteritems():
            if isinstance(v, NameField):
                if v.name is None:
                    v.name = k
                fm.add_field(v)
            elif isinstance(v, FileExtension):
                fm.set_ext(v)
            else:
                retdict[k] = v
        fm.finalize()
        retdict['_meta'] = fm
        return type.__new__(cls, name, bases, retdict)


class FileName(object):
    __metaclass__ = FileNameBase

    def __init__(self, _s=None, **kwargs):
        self._meta.init_fnameobj(self)
        if _s:
            d = self._meta.parse_string(_s)
            for k, v in d.iteritems():
                self._check_key(k)
                setattr(self, k, v)
        else:
            for k, v in kwargs.iteritems():
                self._check_key(k)
                setattr(self, k, v)

    def _check_key(self, k):
        if not hasattr(self, k):
            raise AttributeError, ("%s has no attribute '%s'."
                                   % (self.__class__.__name__, k))

    def to_string(self):
        return self._meta.to_string(self)

    def get_extension(self):
        return str(self._meta.file_ext)
