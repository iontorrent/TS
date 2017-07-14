#!/usr/bin/python
# Copyright 2017 Thermo Fisher Scientific. All Rights Reserved.

from fabric.api import local, lcd

RND_PLUGINS = [
    "bubblePlots",
]
GIT_TAG = "master"
GIT_DIR = "rndplugins"
GIT_URL = "ssh://git@stash.amer.thermo.com:7999/ts/rndplugins.git"


def update_plugins():
    local("rm -rf %s" % GIT_DIR)
    local("mkdir -p %s" % GIT_DIR)
    with lcd(GIT_DIR):
        for plugin in RND_PLUGINS:
            local("git archive --remote %s %s %s | tar -x" % (GIT_URL, GIT_TAG, plugin))
