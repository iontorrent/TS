#!/usr/bin/env python
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
'''
    Writes a new plugin execution control file, if one does not exist.  Updates
    the format if one already exists.
'''
import os
import sys
import json
# Set to True to get debug print statements
DEBUG = True
PLUGIN_DIR='/results/plugins'

class PlugConfFile:
    def __init__(self):
        self.obj = None
        
    def WriteFile (self,filename):
        '''Writes object to json file'''
        fp = open(filename,'wb')
        json.dump(self.obj,fp, indent=4)
        fp.close()
        
    def ReadFile (self,filename):
        '''Reads json file into object'''
        if DEBUG:
            print "Opening file '%s'" % filename
        try:
            fp = open(filename,'rb')
            try:
                self.obj = json.load(fp)
            finally:
                fp.close()
                
            if DEBUG:
                print "Contents of file:"
                print self.obj
                
            return self.obj
        except IOError:
            print "Could not open %s" % filename
    
    def ListPlugins (self,filepath):
        '''Returns list of plugins found in the plugins directory.
        Plugins are subdirectories found in the plugins directory'''
        self.obj = []
        try:
            flist=os.listdir(filepath)
            flist = sorted(flist,reverse=True)  #reverse alpha-mimics existing
        except:
            print "something went wrong"
            return
        
        cnt = 0
        for f in flist:
            dir = os.path.join(filepath,f)
            if os.path.isdir(dir) == True:
                entry = {'name':f,
                         'version':0,
                         'path':dir,
                         'execute':True,
                         'autorun':True,
                         'order':cnt,
                         'order_directive':None,
                         'chipEnable':['314','316','318','324'],
                         'chipDisable':[]}
                self.obj.append(entry)
                cnt += 1

        if DEBUG:
            for l in self.obj:
                print l
        return self.obj
        
    def ChipIsEnabled (self,chipType, plugin):
        '''Returns whether given plugin should execute for given chip type'''
        if chipType in plugin['chipEnable']:
            return True
        else:
            return False
        
    def ChipIsDisabled (self,chipType, plugin):
        '''Returns whether given plugin should not exeecute for given chip type'''
        if chipType in plugin['chipDisable']:
            return True
        else:
            return False
    
def main(argv):
    if len(argv) != 2:
        print "Option error"
        return 1
    
    thisFile = PlugConfFile()
    
    #Test reading json file
    #obj = thisFile.ReadFile(argv[1])
    #if obj == None:
    #    return 1
    
    #Lets get the list of plugins
    plugins = thisFile.ListPlugins(PLUGIN_DIR)
    if plugins == None:
        return 1
    
    chipIs = '315'
    print "Is %s enabled? " % chipIs
    print thisFile.ChipIsEnabled(chipIs, plugins[0])
    chipIs = '318'
    print "Is %s enabled? " % chipIs
    print thisFile.ChipIsEnabled(chipIs, plugins[0])
    chipIs = '315'
    print "Is %s disabled? " % chipIs
    print thisFile.ChipIsDisabled(chipIs, plugins[0])
    chipIs = '318'
    print "Is %s disabled? " % chipIs
    print thisFile.ChipIsDisabled(chipIs, plugins[0])
    
    #Write the config file
    thisFile.WriteFile("./testconfig")
    
    print "Hooray.  File updated."
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
