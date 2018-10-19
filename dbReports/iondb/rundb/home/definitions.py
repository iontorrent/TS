# Copyright (C) 2017 Ion Torrent Systems, Inc. All Rights Reserved
from iondb.rundb.models import RUNNING_STATES

# Runs stages
LIB_PREP = "Library Prep"
TEMPL_PREP = "Template Prep"
SEQUENCING = "Sequencing"
ANALYSIS = "Analysis"
PLUGINS = "Plugins"

# Runs states
IN_PROGRESS = "In progress"
DONE = "Complete"
ERROR = "Error"

# instrument states
CONNECTED = 'Connected'
ALARM = 'Alarm'
OFFLINE = 'Offline'

# instrument types for icon
INSTRUMENT_TYPES = ['PGM', 'Proton', 'S5']

# Constants to drive the logic in the templates
DASHBOARD_STAGES = {
    LIB_PREP: {
        "index": 0,
        "display_name": "Library Prep",
        "name_label": "Sample Set name",
    },
    TEMPL_PREP: {
        "index": 1,
        "display_name": "Template Prep",
        "name_label": "Planned Run name",
    },
    SEQUENCING: {
        "index": 2,
        "display_name": "Sequencing",
        "name_label": "Run name",
    },
    ANALYSIS: {
        "index": 3,
        "display_name": "Analysis",
        "name_label": "Run name",
        "show_report_url": True,
    },
    PLUGINS: {
        "index": 4,
        "display_name": "Plugins",
        "name_label": "Run name",
        "show_report_url": True,
    },
}

