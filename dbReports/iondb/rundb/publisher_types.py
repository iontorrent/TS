# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
"""
    Constants and labels for Publishers
    2017:
        BED
        refAnnot
"""

# BED publisher
AMPLISEQ = "ampliseq"
TARGET = "target"
HOTSPOT = "hotspot"
SSE = "sse"

BED_TYPES = {
    AMPLISEQ: "Ampliseq ZIP",
    TARGET: "Target Regions",
    HOTSPOT: "Hotspots",
    SSE: "SSE",
}

# refAnnot publisher
ANNOTATION = "Annotation"
AUXILIARYREFERENCE = "AuxiliaryReference"

REFANNOT_TYPES = {ANNOTATION: "Annotation", AUXILIARYREFERENCE: "Auxiliary Reference"}


def get_publisher_types():
    all_types = dict(list(BED_TYPES.items()) + list(REFANNOT_TYPES.items()))
    return all_types
