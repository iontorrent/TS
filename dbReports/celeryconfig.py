# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved

# Name of nodes to start, here we have a single node
CELERYD_NODES="w1"
# or we could have three nodes:
#CELERYD_NODES="w1 w2 w3"

# Where to chdir at start.
CELERYD_CHDIR="/opt/ion"

# How to call "manage.py celeryd_multi"
CELERYD_MULTI="$CELERYD_CHDIR/manage.py celeryd_multi"

# How to call "manage.py celeryd_multi"
CELERYD_CTL="$CELERYD_CHDIR/manage.py celeryctl"

# Extra arguments to celeryd
CELERYD_OPTS="--concurrency=1"


# Name of the celery config module.
CELERY_CONFIG_MODULE="celeryconfig"

# %n will be replaced with the nodename.
CELERYD_LOG_FILE="/var/log/ion/celery_%n.log"
CELERYD_PID_FILE="/var/run/celery/celery_%n.pid"

# Workers should run as an unprivileged user.
CELERYD_USER="root"
CELERYD_GROUP="root"

# Name of the projects settings module.
DJANGO_SETTINGS_MODULE="iondb.settings"
