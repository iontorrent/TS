# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

import json
import struct

import cnumtypes

ctypes_dict = dict([(ele, getattr(cnumtypes, ele)) for ele in dir(cnumtypes)
    if not ele.count('_')])
reverse_ctypes = dict([(v,k) for k,v in ctypes_dict.iteritems()])

def check_vector(fn):
        def ret(obj, *args, **kwargs):
            if obj._vector is None:
                raise TypeError, "%s is not a list field." % obj._name
            else:
                return fn(obj, *args, **kwargs)
        ret.func_name = fn.func_name
        return ret

class BinaryFileData(object):
    def __init__(self, name, fields, meta, own_type):
        self._name = name
        self._meta = meta
        self._fields = fields
        self._progenitor = None
        self._children = {}
        self._vector = None
        self._value_indices = {}
        self._curr_index = 0
        self._own_type = own_type
    def set(self, key, value):
        setattr(self, key, value)
        if key not in self._value_indices:
            self._value_indices[key] = self._curr_index
            self._curr_index += 1
    def get(self, key):
        if key in self._value_indices:
            return getattr(self, key)
        else:
            raise KeyError, "%s %s has no field %s" % (
                self.__class__.__name__, self._name, str(key)
            )
    def ordered_values(self):
        items = self._value_indices.items()
        items.sort()
        for k,v in items:
            yield k,self.get(k)
    def ordered_list(self):
        return [ele for key,ele in self.ordered_values()]
    def to_python_serial(self, list_limit=None):
        ret = {}
        for k,v in self.ordered_values():
            if isinstance(v, BinaryFileData) and not v._vector:
                ret[k] = v.to_python_serial(list_limit)
            elif hasattr(v, '__len__') and not isinstance(v,str):
                    arr = []
                    for ndx,ele in enumerate(v):
                        if ndx == list_limit:
                            break
                        if ele == self:
                            exit(-1)
                        if isinstance(ele, BinaryFileData):
                            #print "descending on",k,"to",type(ele._progenitor),ele._name
                            arr.append(ele.to_python_serial(list_limit))
                        else:
                            arr.append(ele)
                    ret[k] = arr
            else:
                ret[k] = v
        return ret
    def __contains__(self, key):
        return key in self._value_indices
    @check_vector
    def __getitem__(self, ndx):
        return self._vector[ndx]
    #@check_vector
    def __len__(self):
        if self._vector is None:
            return True
        return len(self._vector)
    @check_vector
    def add_default_child(self):
        ret = self._own_type.default_value()
        #prog_name = self._progenitor.name
        #if prog_name in self._value_indices:
        #    arr = self.get(prog_name)
        #else:
        #    arr = []
        #    self.set(prog_name, arr)
        #arr.append(ret)
        self._vector.append(ret)
        return ret
    def __bool__(self):
        return True
  
class BinaryFileWrapper(object):
    def __init__(self, name, bftype, meta, own_type):
        self.name = name
        self.fields = meta.fields
        self.bftype = bftype
        self._own_type = own_type
        self._meta = meta
    def default_value(self):
        #ret = self.bftype(fh=None)
        ret = BinaryFileData("sub_%s" % self.name, self.fields, self._meta, self._own_type)
        vals = []
        for f in self.fields:
            dval = f.default_value()
            ret.set(f.name, dval)
        return ret
    
class Endianness(object):
    _symbol = "="
    _name = "Standard Byte Order"
    @classmethod
    def __str__(cls):
        return cls._name

class LittleEndian(Endianness):
    _symbol = "<"
    _name = "Little Endian"
    
class BigEndian(Endianness):
    _symbol = ">"
    _name = "Big Endian"
    
class NetworkEndian(Endianness):
    _symbol = "!"
    _name = "Network Byte Order"
    
class NativeEndian(Endianness):
    _symbol = "@"
    _name = "Native Byte Order"
        
class BinaryFileField(object):
    def __init__(self, rank):
        self.name = None
        self.rank = rank
        self.pre_func = None
        self.post_func = None
    def set_pre_func(self, f):
        self.pre_func = f
    def set_post_func(self, f):
        self.post_func = f
    def set_name(self,name):
        self.name = name
    def get_header_fmt(self):
        return None
    def get_body_fmt(self, header_data, query):
        return None
    def get_children(self, body_data, query):
        return None
    def children_as_list(self):
        return False
    def to_python(self, data):
        return None
    def from_python(self, data, childfields):
        return [], ''
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.__str__())
    def pad_to_mod(self):
        return None
    def default_value(self):
        return None
    def get_child_type(self):
        return None
    def body_oracle_query(self):
        return None,None
    def child_oracle_query(self):
        return None,None
    
class GroupLength(object):
    def __init__(self, field, include=None, exclude=None):
        self.include = include
        self.exclude = exclude
        self.field = field
    def calcsize(self, sizedict):
        ret = 0
        if self.exclude is not None:
            ex = set(self.exclude)
        else:
            ex = None
        if self.include:
            for k in self.include:
                ret += sizedict[k]
        else:
            for k,v in sizedict.iteritems():
                if not ex or k not in ex:
                    ret += sizedict[k]
        return ret
    
class SingleField(BinaryFileField):
    def __init__(self, rank, dtype):
        if dtype in ctypes_dict:
            self.dtype = ctypes_dict[dtype]
            self.dtype_name = dtype
        else:
            self.dtype = dtype
            self.dtype_name = reverse_ctypes[dtype]
        BinaryFileField.__init__(self, rank)
    def get_body_fmt(self, header_data,query):
        if header_data:
            raise ValueError, ("No header data expected, but got '%s'."
                    % header_data)
        return self.dtype
    def to_python(self, data):
        return data[0]
    def from_python(self, data, childfields):
        return data, self.dtype
    def default_value(self):
        return cnumtypes.default_type_values[self.dtype]

class MultiField(SingleField):
    def __init__(self, rank, dtype, n):
        self.n = int(n)
        SingleField.__init__(self, rank, dtype)
        self.fmt = "%d%s" % (self.n, self.dtype)
    def get_body_fmt(self, header_data,query):
        return self.fmt
    def to_python(self, data):
        return data
    def from_python(self, data, childfields):
        return data, self.fmt
    def default_value(self):
        val = SingleField.default_value(self)
        return [val]*self.n

class StringField(MultiField):
    def __init__(self, rank, n):
        MultiField.__init__(self, rank, cnumtypes.char, n)
        self.pre_func = lambda data: ''.join(data)
        self.post_func = lambda data: list(data)

class MagicNumberField(StringField):
    def __init__(self, rank, n, expected):
        StringField.__init__(self, rank, n)
        self.expected = expected
        self.old_pre_func = self.pre_func
        def check(data):
            data = self.old_pre_func(data)
            if data != self.expected:
                raise ValueError, ("Expected magic number '%s' but got '%s'."
                        % (self.expected, data))
            return data
        self.pre_func = check
    def default_value(self):
        return self.expected

class RepeatedSingleField(SingleField):
    def __init__(self, rank, dtype, index_dtype):
        self.hdr = SingleField(rank=0, dtype=index_dtype)
        SingleField.__init__(self, rank, dtype)
    def fmt_n(self,n):
        return "%d%s" % (n,self.dtype)
    def get_header_fmt(self):
        return self.hdr.get_body_fmt(None)
    def get_body_fmt(self, header_data,query):
        return self.fmt_n(int(header_data[0]))
    def to_python(self, data):
        return data
    def from_python(self,data,childfields):
        return [len(data), data], self.hdr.dtype + self.fmt_n(len(data))
    def default_value(self):
        return []
                
class ParentField(BinaryFileField):
    _EMPTY = []
    def __init__(self, rank, index_dtype, child_type, child_args):
        self.hdr = SingleField(index_dtype)
        self.child_type = child_type
        self.child_args = child_args
        self.nchildren = None
        self.children = None
        BinaryFileField.__init__(self, rank)
    def get_body_fmt(self, header_data,query):
        return self.hdr.get_body_fmt(None)
    def get_children(self, body_data, query):
        nchildren = int(body_data[0])
        ret = []
        for i in range(nchildren):
            ret.append(self.child_type(*self.child_args))
        return ret
    def to_python(self,data):
        raise TypeError, "Tried to call to_python on field with children."
    def from_python(self,data,childfields):
        return [len(childfields)], self.hdr.dtype
    def get_child_type(self):
        return self.child_type
      
class SubfileField(BinaryFileField):
    def __init__(self, rank, subfile):
        self.subfile = subfile
        BinaryFileField.__init__(self, rank)
    def get_children(self, body_data, query):
        return self.subfile.data#._fields
    def get_child_type(self):
        return self.subfile
    
class LookupRepeatedSubfileField(SubfileField):
    def __init__(self, rank, subfile, hdr_field_name):
        self.hdr_field_name = hdr_field_name
        SubfileField.__init__(self, rank, subfile)
    def child_oracle_query(self):
        return self.hdr_field_name, FieldManager.Oracle.LENGTH
    def get_children(self, body_data, query):
        n = int(query)
        ret =  [SubfileField(i,self.subfile) for i in range(n)]
        for sf in ret:
            sf.set_name("%s_sub_%d" % (self.name, sf.rank))
        return ret
    def children_as_list(self):
        return True
        
class LookupRepeatedField(SingleField):
    def __init__(self, rank, dtype, hdr_field_name):
        self.hdr_field_name = hdr_field_name
        SingleField.__init__(self, rank, dtype)
    def fmt_n(self, n):
        return "%d%s" % (n, self.dtype)
    def body_oracle_query(self):
        return self.hdr_field_name, FieldManager.Oracle.LENGTH
    def get_body_fmt(self, header_data, query):
        n = int(query)
        return self.fmt_n(n)
    def to_python(self, data):
        return data
    def from_python(self, data, childfields):
        return data, self.fmt_n(len(data))
    def default_value(self):
        return []
        
class LookupStringField(LookupRepeatedField):
    def __init__(self, rank, hdr_field_name):
        LookupRepeatedField.__init__(self, rank, cnumtypes.char, hdr_field_name)
        self.set_pre_func(''.join)
        self.set_post_func(tuple)
    def default_value(self):
        return ''
 
class LookupRepeatedFixedPointField(LookupRepeatedField):
    def __init__(self, rank, dtype, hdr_field_name, decpoints):
        self.decpoints = decpoints
        self.postmult = float(10**decpoints)
        self.premult = 1.0/self.postmult
        LookupRepeatedField.__init__(self,rank,dtype,hdr_field_name)
        self.set_pre_func(lambda x: [self.premult*float(ele) for ele in x])
        self.set_post_func(lambda x: [int(self.postmult*ele) for ele in x])
           
class Padding(BinaryFileField):
    def __init__(self, rank, mod):
        BinaryFileField.__init__(self, rank)
        self.mod = mod
    def pad_to_mod(self):
        return self.mod
    

class FieldManager(object):
    class Oracle(object):
        LENGTH = 'l'
        INFO = 'i'
        def __init__(self, objects):
            self.objects = objects
        def get_initial_context(self, requests):
            first = requests[0]
            objlen = len(self.objects)
            if first[:2] == '..':
                ndx = objlen - 2
                start = 1
            elif first[0] == '.':
                ndx = objlen - 1
                start = 1
            else:
                ndx = 0
                start = 0
            #print "INDEX:", ndx, "FIRST:[%s]" % first
            return self.objects[ndx],ndx,start
        def find(self, key):
            requests = key.split('/')
            target,ndx,start = self.get_initial_context(requests)
            for r in requests[start:-1]:
                if r == '..':
                    ndx -= 1
                    if ndx < 0:
                        raise ValueError, "Cannot rise above current level."
                    target = self.objects[ndx]
                else:
                    target = target.get(r)
            return target, requests[-1]
        def get(self, key, reqtype):
            if reqtype is None and key is None:
                return None
            target, lastreq = self.find(key)
            return target.get(lastreq)
        def set(self, key, value):
            target, lastreq = self.find(key)
            target.set(lastreq, value)
    def __init__(self, fields, grouplengths, grouplengthfields, endianness):
        self.fields = fields
        self.endian = endianness
        self.grouplengths = grouplengths
        self.glfields = grouplengthfields
    def make_defaults(self, python_obj):
        return self._make_defaults(self.fields, python_obj)
    def _make_defaults(self, fields, python_obj):
        for f in fields:
            #if not f.children_as_list():
            child_type = f.get_child_type()
            if child_type is not None:
                python_obj._children[f.name] = child_type
                if issubclass(child_type, BinaryFile):
                    fields = child_type.data._meta.fields
                    child_meta = child_type.data._meta
                    obj_field_type = BinaryFileWrapper(f.name, child_type,
                            child_meta, python_obj._own_type)
                    obj = obj_field_type
                else:
                    fields = [child_type]
                    child_meta = self
                    obj_field_type = child_type                
                obj = self.make_python_obj(f.name, fields, obj_field_type)
                obj._progenitor = BinaryFileWrapper(f.name, child_type,
                            child_type.data._meta, python_obj._own_type)
                if f.children_as_list():
                    toset = self.make_python_obj(f.name, child_type, obj_field_type)
                    toset._vector = obj.ordered_list()
                else:
                    toset = obj
            else:
                toset = f.default_value()
            python_obj.set(f.name, toset)
    def load(self, python_obj, npt_buf):
        self._load(self.fields, python_obj, npt_buf, 0, [python_obj], None)        
    def _load(self, fields, python_obj, buf, bufndx, parent, name):
        oracle = self.Oracle(parent)
        retlist = []
        sizes = {}
        for f in fields:
            nbytes = 0
            if not isinstance(f, BinaryFileField):
                raise TypeError, ("Expected BinaryFileField, got '%s'."
                        % str(type(f)))
            header_fmt = f.get_header_fmt()
            if header_fmt is not None:
                nbytes += struct.calcsize(header_fmt)
                bufndx,header_data = self.fmt_to_data(header_fmt, buf, bufndx)
            else:
                header_data = None
            req,reqtype = f.body_oracle_query()
            queryout = oracle.get(req,reqtype)
            body_fmt = f.get_body_fmt(header_data, queryout)
            if body_fmt is not None:
                body_fmt_size = struct.calcsize(body_fmt)
                if body_fmt_size > 1000:
                    print f.name, req, queryout, body_fmt, [ele for ele in python_obj.ordered_values()]
                nbytes += body_fmt_size
                bufndx,body_data = self.fmt_to_data(body_fmt, buf, bufndx)
            else:
                body_data = None
            padmod = f.pad_to_mod()
            if padmod:
                while bufndx % padmod:
                    bufndx += 1
                    nbytes += 1
            req,reqtype = f.child_oracle_query()
            queryout = oracle.get(req,reqtype)
            children = f.get_children(body_data, queryout)
            if children is not None:
                if isinstance(children, BinaryFileData):
                    cmeta = children._meta
                    cdata = children
                    bufndx = cmeta._load(cmeta.fields, cdata, buf,
                            bufndx, parent + [cdata], f.name)
                    obj = cdata
                    toset = obj
                else:
                    ctype = f.get_child_type()
                    if issubclass(ctype, BinaryFile):
                        ctype = BinaryFileWrapper(f.name, ctype,
                                ctype.data._meta, python_obj._own_type)
                    obj = self.make_python_obj(f.name, children, ctype)
                    obj._progenitor = f
                    python_obj._children[f.name] = children
                    python_obj._child_type = ctype
                    if f.children_as_list():
                        subparent = parent + [python_obj]
                    else:
                        subparent = parent + [obj]
                    oldbufndx = bufndx
                    bufndx = self._load(children,obj,buf,bufndx,subparent,f.name)
                    nbytes += bufndx - oldbufndx
                    if f.children_as_list():
                        toset = self.make_python_obj(f.name, children, ctype)
                        toset._vector = obj.ordered_list()
                    else:
                        toset = obj
            else:
                final_data = f.to_python(body_data)
                if f.pre_func is not None:
                    final_data = f.pre_func(final_data)
                toset = final_data
            #if not children or (children and f.children_as_list()):
            python_obj.set(f.name, toset)
            sizes[f.name] = nbytes
        return bufndx
    def fmt_to_data(self, fmt, buf, pos):
        fmt = self.endian._symbol + fmt
        nextpos = pos + struct.calcsize(fmt)
        data = struct.unpack_from(fmt,buf,pos)
        return nextpos,data
    def _save(self, python_obj, buf, fmtlist, length):
        sizes = {}
        size_indices = {}
        #compile a list of the fields that need to be replaced
        fields = python_obj._fields
        for f in fields:
            start_index = len(buf)
            nbytes = 0
            name = f.name
            val = python_obj.get(name)
            if name in python_obj._children:
                children = python_obj._children[name]
            else:
                children = None
            if f.post_func is not None:
                val = f.post_func(val)
            raw,fmt = f.from_python(val, children)
            buf.append(raw)
            fmtlen = struct.calcsize(fmt)
            length += fmtlen
            nbytes += fmtlen
            fmtlist.append(fmt)
            padmod = f.pad_to_mod()
            if padmod:
                fmtapp = []
                count = 0
                padsize = struct.calcsize(cnumtypes.pad)
                while length % padmod:
                    count += 1
                    length += padsize
                    nbytes += padsize
                fmtlist.append(cnumtypes.pad*count)
            child_sizes = None
            if children is not None and f.children_as_list():
                for obj in val:
                    if isinstance(obj, BinaryFileData):
                        child_sizes = obj._meta._save(obj,buf,fmtlist, length)
                    else:
                        child_sizes = self._save(obj, buf, fmtlist, length)
            
            sizes[f.name] = nbytes
            if f.name in self.glfields:
                diff = len(buf) - start_index
                if diff != 1:
                    raise ValueError, ("Expected one list index for GroupLength"
                            " field, but got %d." % diff)
                size_indices[f.name] = start_index
        for k,v in self.grouplengths.iteritems():
            #print sizes
            if k in size_indices:
                size = v.calcsize(sizes)
                buf[size_indices[k]] = size               
        return sizes
    def save(self, python_obj):
        buf = []
        fmt = [self.endian._symbol]
        self.set_length_fields(python_obj)
        self._save(python_obj, buf, fmt, 0)
        fmt = ''.join(fmt)
        outbuf = []
        self.flatten(buf,outbuf)
        return struct.pack(fmt, *outbuf)

    def set_length_fields(self, python_obj):
        self._set_length_fields(python_obj, self.fields, [python_obj])
    def _set_length_fields(self, python_obj, fields, parent):
        def handle_req(req,typ,debug_caller=None):
            if typ == self.Oracle.LENGTH:
                #print req, debug_caller.name, repr(debug_caller)
                #print oracle.get(req,0), val
                #print [ele for ele in python_obj.ordered_values()]
                #print '* * *'
                #return
                oracle.set(req,len(val))
        oracle = self.Oracle(parent)
        for f in fields:
            val = python_obj.get(f.name)
            breq,btype = f.body_oracle_query()
            handle_req(breq,btype,f)
            creq,ctype = f.child_oracle_query()
            handle_req(creq,ctype,f)
            if isinstance(val, BinaryFileData) and val._vector is None:
                subparent = parent + [val]
                val._meta._set_length_fields(val, val._meta.fields, subparent)
            elif hasattr(val, '__len__'):
                for ele in val:
                    if isinstance(ele, BinaryFileData):
                        subparent = parent + [ele]
                        ele._meta._set_length_fields(ele,ele._meta.fields,subparent)
    @classmethod
    def flatten(cls, inbuf, outbuf):
        for ele in inbuf:
            if isinstance(ele, BinaryFileData):
                raise TypeError
            if isinstance(ele,list) or isinstance(ele,tuple):
                cls.flatten(ele, outbuf)
            elif isinstance(ele,str):
                outbuf.extend(ele)
            elif isinstance(ele,unicode): # Struct.pack has problems with unicode, so we str it.
                outbuf.extend(str(ele))
            else:
                outbuf.append(ele)
    def make_python_obj(self, name, fields, obj_field_type):
        return BinaryFileData(name, fields, self, obj_field_type)
    
class BinaryFileBase(type):
    def __new__(cls, name, bases, dct):
        fields = {}
        grouplengths = {}
        grouplengthfields = {}
        retdict = {}
        end = None
        end_name = None
        for k,v in dct.iteritems():
            if isinstance(v, BinaryFileField):
                v.set_name(k)
                fields[v.rank] = v
            elif isinstance(v, Endianness):
                if end is not None:
                    raise AttributeError, "More than one endianness defined."
                else:
                    end = v
                    end_name = k
            elif isinstance(v, GroupLength):
                bff = v.field
                bff.set_name(k)
                fields[bff.rank] = bff
                grouplengths[k] = v
                grouplengthfields[bff.name] = v
            else:
                retdict[k] = v
        if end is None:
            end = Endianness
            end_name = "endianness"
        fields = fields.items()
        fields.sort()
        fields = [b for a,b in fields]
        fm = FieldManager(fields, grouplengths, grouplengthfields, end)
        retdict[end_name] = str(v)
        datatype = BinaryFileWrapper('top', cls, fm, None)
        data = BinaryFileData('data', fields, fm, datatype)
        retdict['data'] = data
        return type.__new__(cls, name, bases, retdict)
        
class BinaryFile(object):
    __metaclass__ = BinaryFileBase
    def __init__(self, fh=None):
        if fh is not None:
            if hasattr(fh, 'read') and callable(fh.read):
                self.load(fh)
            else:
                self.loads(fh)
        else:
            self.data._meta.make_defaults(self.data)
    def load(self, fh):
        self.loads(fh.read())
    def loads(self, buf):
        self.data._meta.load(self.data, buf)
    def dump(self, fh):
        fh.write(self.dumps())
    def dumps(self):
        return self.data._meta.save(self.data)
    def to_python_serial(self, list_limit=None):
        self.data._meta.set_length_fields(self.data)
        return self.data.to_python_serial(list_limit)
    def to_json(self, list_limit=None):
        pyserial = self.to_python_serial(list_limit)
        return json.dumps(pyserial, indent=4)
    
from cnumtypes import *    
