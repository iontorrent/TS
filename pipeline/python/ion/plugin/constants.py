# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

"""
Definitions for enum like constants
"""

__all__ = ("Feature", "RunType", "RunLevel", "lookupEnum", "runLevelsList")


def enum(**enums):
    return type("Enum", (), enums)


# Helper lookup method - move to constants?


def lookupEnum(enum, item):
    for (k, v) in enum.__dict__.items():
        if item == v:
            return k
    return None


def runLevelsList():
    # returns runlevels in the order launched by the pipeline
    return [
        RunLevel.PRE,
        RunLevel.BLOCK,
        RunLevel.SEPARATOR,
        RunLevel.DEFAULT,
        RunLevel.POST,
        RunLevel.LAST,
    ]


Feature = enum(EXPORT="export")

RunType = enum(COMPOSITE="composite", THUMB="thumbnail", FULLCHIP="wholechip")

RunLevel = enum(
    PRE="pre",
    DEFAULT="default",
    BLOCK="block",
    POST="post",
    SEPARATOR="separator",
    LAST="last",
)
