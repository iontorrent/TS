# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
# Cluster settings for Torrent Suite Compute nodes

PLUGINSERVER_HOST = 'localhost'
PLUGINSERVER_PORT = 9191
JOBSERVER_HOST = 'localhost'
JOBSERVER_PORT = 10000
CRAWLER_HOST = 'localhost'
CRAWLER_PORT = 10001

# import from the local settings file
try:
    from local_cluster_settings import *
except ImportError:
    pass
