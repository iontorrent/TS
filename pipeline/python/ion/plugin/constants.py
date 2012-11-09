# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

"""
Definitions for enum like constants
"""

__all__ = ('Feature', 'RunType', 'RunLevel')

def enum(**enums):
        return type('Enum', (), enums)

Feature = enum(
    EXPORT='export',
)

RunType = enum(
    COMPOSITE='composite',
    THUMB='thumbnail',
    FULLCHIP='wholechip'
)

RunLevel = enum(
    PRE='pre',
    DEFAULT='default',
    BLOCK='block',
    POST='post',
    SEPARATOR='separator',
    LAST='last',
)

