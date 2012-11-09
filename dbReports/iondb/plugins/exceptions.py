# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

class IonPluginError(Exception):
    pass

class PluginNotFound(IonPluginError):
    def __init__(self, name, **kwargs):
        super(PluginNotFound, self).__init__(name, **kwargs)
        self.name=name
    def __repr__(self):
        return "Plugin %s" % name + repr(self.args)

class XMLRPCError(IonPluginError):
    pass


