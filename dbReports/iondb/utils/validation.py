# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
import re

class Rules():
    REQUIRED = 'required' # input required
    INT = 'int'           # integer
    UINT = 'uint'         # integer >= 0
    ALPHANUM = 'alphanum' # letters and numbers
    NAME = 'name'         # letters, numbers, space, and .-_
    SLUG = 'slug'         # letters, numbers, underscores or hyphens
    LEAD_CHAR = 'leadchar' # lead character must be letter or number
    MIN_LENGTH = 'min_l'  # min length
    MAX_LENGTH = 'max_l'  # max length
    NUC = 'nuc'           # ATCG
    KEYWORDS = "keywords" # compare value to a fixed list of supported values 
    NOT_FOUND = "not found"  #input not recognized
    
    @classmethod
    def get_error(cls, rule, arg=''):
        msg = {
            cls.REQUIRED: ' is required',
            cls.INT     : ' must be an integer',
            cls.UINT    : ' must be a positive integer',
            cls.ALPHANUM: ' should contain only numbers and letters (without special characters)',
            cls.NAME    : ' should contain only letters, numbers, spaces, and the following: . - _ ',
            cls.SLUG    : ' should contain only letters, numbers, underscores or hyphens',
            cls.LEAD_CHAR:' should start with letter or number',
            cls.MIN_LENGTH: ' should be at least %s characters long',
            cls.MAX_LENGTH: ' should not be longer than %s characters',
            cls.NUC     : ' valid values are TACG',
            cls.KEYWORDS : ' valid values are ',
            cls.NOT_FOUND : " not found for %s ",
            'unknown' : ' failed to validate. Validator function "%s" is not defined'
        }
        if rule in msg:
            if rule == cls.MIN_LENGTH or rule == cls.MAX_LENGTH or rule == cls.NOT_FOUND:
                return msg[rule] % arg
            elif rule == cls.KEYWORDS:
                return msg[rule] + ", ".join(arg)
            else:
                return msg[rule]
        else:
            return msg['unknown'] % rule

    @classmethod
    def validate(cls, rule, value, arg=''):
        if rule == cls.REQUIRED:
            if isinstance(value, str):
                value = value.strip()
            if value:
                return True
        elif rule == cls.INT:
            try:
                int(value)
                return True
            except ValueError:
                pass
        elif rule == cls.UINT:
            try:
                return int(value) > 0
            except ValueError:
                pass
        elif rule == cls.ALPHANUM:
            return bool(re.match('^$|[\w]+$', value) )
        elif rule == cls.NAME:
            return bool(re.match('^$|[a-zA-Z0-9\\-_. ]+$', value) )
        elif rule == cls.SLUG:
            return bool(re.match('^$|[a-zA-Z0-9\\-_]+$', value) )
        elif rule == cls.LEAD_CHAR:
            return bool(re.match('^$|[a-zA-Z0-9]$', value.strip()[0]) )
        elif rule == cls.MIN_LENGTH:
            return len(value.strip()) >= arg
        elif rule == cls.MAX_LENGTH:
            return len(value.strip()) <= arg
        elif rule == cls.NUC:
            return bool(re.match('^$|[atcgATCG]+$', value) )
    
        return False


''' Validation helper functions '''

def is_valid_chars(value):
    # only letters, numbers, spaces, dashes, underscores and dots
    return Rules.validate(Rules.NAME, value)

def invalid_chars_error(displayedName):
    return displayedName + Rules.get_error(Rules.NAME)

def is_valid_leading_chars(value):
    # leading chars must be letters or numbers
    if value:
        return Rules.validate(Rules.LEAD_CHAR, value)
    return True

def invalid_leading_chars(displayedName):
    return displayedName + Rules.get_error(Rules.LEAD_CHAR)

def is_valid_length(value, maxLength):
    # value length must be within the maximum allowed
    if value:
        return Rules.validate(Rules.MAX_LENGTH, value, maxLength)
    return True

def invalid_length_error(displayedName, maxLength):
    return displayedName + Rules.get_error(Rules.MAX_LENGTH, maxLength)


def invalid_keyword_error(displayedName, valid_keyword_list):
    return displayedName + Rules.get_error(Rules.KEYWORDS, valid_keyword_list)


def invalid_not_found_error(displayedName, input):
    return displayedName + Rules.get_error(Rules.NOT_FOUND, input)

    
def is_valid_uint(value):
    # must be non-negative integer
    return Rules.validate(Rules.UINT, value)

def invalid_uint(displayedName):
    return displayedName + Rules.get_error(Rules.UINT)

def has_value(value):
    # value is required
    return Rules.validate(Rules.REQUIRED, value)

def required_error(displayedName):
    return displayedName + Rules.get_error(Rules.REQUIRED)


def is_valid_keyword(value, valid_keyword_list):
    return value.lower() in (key.lower() for key in valid_keyword_list)
    