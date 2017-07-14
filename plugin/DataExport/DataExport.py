#!/usr/bin/env python
# Copyright (C) 2015 Ion Torrent Systems, Inc. All Rights Reserved
# DataExport plugin
#
# This plugin will transfer the associated data file categories to a server.
import os
import sys
import json
import urllib
import traceback
import subprocess
from ion.plugin import PluginCLI, IonPlugin, RunLevel


class DataExport(IonPlugin):
    """
    This plugin automates a manual Data Management Export Action
    """
    version = '5.4.0.1'
    author = "bernard.puc@thermofisher.com and samuel.thoraval@thermofisher.com"
    runlevels = [RunLevel.LAST]

    # Copied from dmactions.py and dmactions_import.py
    ARCHIVE = 'archive'
    EXPORT = 'export'
    DELETE = 'delete'
    TEST = 'test'
    SIG = 'Signal Processing Input'
    BASE = 'Basecalling Input'
    OUT = 'Output Files'
    INTR = 'Intermediate Files'
    FILESET_TYPES = [SIG, BASE, OUT, INTR]

    def __init__(self):
        IonPlugin.__init__(self)
        self.categories = []
        self.result_pk = ''
        self.report_name = ''
        self.output_dir = ''
        self.dest_dir = ''


    # main method of plugin execution
    def launch(self, data=None):

        print "==============================================================================="

        # Configuration and Instance pages will have
        # username
        #username = 'ionadmin'
        # password
        #password = 'iongenomes'
        # server address
        #server = 'enderman.ite'
        # destination directory
        #dest_dir = '/ion-data'
        #dest_dir = '/media/77f2a9df-7668-41ba-8f84-c567f8418eb0'
        # Boolean flag for each category
        # - Signal Processing Input
        # - Basecalling Input
        # - Output Files
        # - Intermediate Files
        #data_categories = [
        #    {self.SIG:False},
        #    {self.BASE:True},
        #    {self.OUT:True},
        #    {self.INTR:False},
        #]

        try:
            # Get values from plugin environment
            with open('startplugin.json', 'r') as fh:
                spj = json.load(fh)
                self.result_pk      = spj['runinfo']['pk']
                self.output_dir     = spj['runinfo']['results_dir']
                self.report_name    = os.path.basename(spj['runinfo']['report_root_dir'])
                self.dest_dir       = spj['pluginconfig'].get('upload_path', '')
                sigproc             = spj['pluginconfig'].get('sigproc', 'off')
                basecalling         = spj['pluginconfig'].get('basecalling', 'off')
                output              = spj['pluginconfig'].get('output', 'off')
                intermediate        = spj['pluginconfig'].get('intermediate', 'off')
        except Exception:
            # something unexpected.
            print traceback.format_exc()
            self.plugin_exception(traceback.format_exc())
            return True

        data_categories = [
            {self.SIG:False if sigproc == 'off' else True},
            {self.BASE:False if basecalling == 'off' else True},
            {self.OUT:False if output == 'off' else True},
            {self.INTR:False if intermediate == 'off' else True},
        ]

        #Input validation
        if self.dest_dir == "":
            # Error - probably global config has not been completed by user.
            print "Upload path has not been configured. Did you run the global configuration for this plugin?"
            self.plugin_not_configured_error()
            return True

        # Action
        # Generate URL to post that triggers manual DM action
        #
        for item in data_categories:
            if item.values()[0]:
                self.categories.append(item.keys()[0])

        commentary = self.seek_commentary()

        raw_post_data = {
            'backup_dir': self.dest_dir,
            'confirmed': True,
            'categories': self.categories,
            'comment': commentary,
        }

        if len(self.categories):
            #===============================================
            # Copying to local directory
            #===============================================
            self.generate_completion_html()
            print "Uploading %s" % (self.report_name)
            print "File categories: %s" % (self.categories)
            print "Destination: %s" % (self.dest_dir)
            error_msg = []
            connection_url = "http://localhost/data/datamanagement/dm_actions/%s/%s/" % (self.result_pk, self.EXPORT)
            try:
                conn = urllib.urlopen(connection_url, json.dumps(raw_post_data))
            except IOError:
                print 'could not make connection %s' % connection_url
                try:
                    connection_url = 'https://localhost/data/datamanagement/dm_actions/%s/%s/' % (self.result_pk, self.EXPORT)
                    conn = urllib.urlopen(connection_url, json.dumps(raw_post_data))
                except IOError:
                    error_msg.append(" !! Failed to submit URL.  could not connect to %s" % connection_url)
                    for line in traceback.format_exc().splitlines():
                        error_msg.append(line)
                    print error_msg
                    self.generate_completion_html(stat_line='Failure', error_msg=error_msg)
                    conn = None

            if conn:
                error_code = conn.getcode()
                if error_code is not 200:
                    error_msg.append(" !! URL failed with error code %d for %s" % (error_code, conn.geturl()))
                    for line in conn.readlines():
                        error_msg.append(line)
                    print error_msg
                    self.generate_completion_html(stat_line='Failure', error_msg=error_msg)
                else:
                    self.generate_completion_html()
        else:
            print "Nothing to do.  No categories were selected to transfer"
            self.generate_completion_html(stat_line="No files were selected")

        # Exit the launch function; exit the plugin
        print "==============================================================================="
        print commentary
        return True


    def get_files_to_transfer(self):
        raw_post_data = {
            'backup_dir': self.dest_dir,
            'confirmed': True,
            'categories': self.categories,
            'comment': "Wildfires",
        }
        connection_url = "http://localhost/data/datamanagement/dm_list_files/%s/%s/" % (self.result_pk, self.EXPORT)
        try:
            conn = urllib.urlopen(connection_url, json.dumps(raw_post_data))
            file_payload = json.loads(conn.read())
        except IOError:
            print traceback.format_exc()

        return file_payload


    def generate_completion_html(self, stat_line="Started", **kwargs):
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print "Could not write status report"
            print traceback.format_exc()
            raise

        msg = "%s" % (self.dest_dir)
        _status_msg = "Progress is not available here.  Look for a message banner at the top of any webpage indicating that the action was scheduled, or completed"
        display_fs.write("<html><head>\n")
        #display_fs.write("<link href=\"/pluginMedia/IonCloud/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body>\n")
        display_fs.write("<bold><h2>DATA EXPORT</h2></bold>")
        #display_fs.write("<p>REPORT NAME: %s</p>" % (self.report_name))
        #display_fs.write("<p>REPORT ID: %s</p>" % (self.result_pk))
        display_fs.write("<p>FILE CATEGORIES:</p>" % (self.categories))
        display_fs.write("<ul>")
        for entry in self.categories:
            display_fs.write("<li>%s</li>" % (entry))
        display_fs.write("</ul>")
        display_fs.write("<p>DESTINATION: %s</p>" % (msg))
        if stat_line not in ["Started", "Completed"]:
            display_fs.write("<p>STATUS: <font color=\"red\">%s</font></p>" % stat_line)
        else:
            display_fs.write("<p>STATUS: %s</p>" % stat_line)
            display_fs.write("<small>%s</small>" % _status_msg)
        for line in kwargs.get('error_msg', []):
            display_fs.write("%s<br>" % line)
        display_fs.write("<hr />")
        display_fs.write("<small>Contact: %s Version: %s</small>" % (self.author, self.version))
        display_fs.write("</body></html>\n")
        display_fs.close()


    def plugin_not_configured_error(self):
        print "Function: %s()" % sys._getframe().f_code.co_name
        #print report
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print "Could not write status report"
            print traceback.format_exc()
            raise

        display_fs.write("<html><head>\n")
        display_fs.write("<link href=\"/pluginMedia/IonCloud/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body>\n")
        display_fs.write("<center>\n")
        display_fs.write("<p> %s </p>" % "PLUGIN IS NOT CONFIGURED.")
        display_fs.write("<p> %s </p>" % "Run the global configuration for this plugin from the <a href=\"/configure/plugins\" target=\"_blank\">Plugins page</a>.")
        display_fs.write("<hr />")
        display_fs.write("<small>Contact: %s Version: %s</small>" % (self.author, self.version))
        display_fs.write("</center></body></html>\n")
        display_fs.close()


    def plugin_exception(self, _exception):
        print "Function: %s()" % sys._getframe().f_code.co_name
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print "Could not write status report"
            print traceback.format_exc()
            raise

        display_fs.write("<html><head>\n")
        display_fs.write("<link href=\"/pluginMedia/IonCloud/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body>\n")
        display_fs.write("<center>\n")
        display_fs.write("<p> %s </p>" % "PLUGIN CODE EXCEPTION")
        display_fs.write("</center>\n")
        for line in _exception.splitlines():
            display_fs.write("%s</br>" % line)
        display_fs.write("<hr />")
        display_fs.write("<small>Contact: %s Version: %s</small>" % (self.author, self.version))
        display_fs.write("</body></html>\n")
        display_fs.close()


    def seek_commentary(self):
        try:
            p1 = subprocess.Popen(["/usr/games/fortune", "-s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, _ = p1.communicate()
            myfortune = stdout
        except:
            myfortune = "Fools like us"
            print traceback.format_exc()
        finally:
            return myfortune


    def report(self):
        pass

# dev use only - makes testing easier
if __name__ == "__main__":
    PluginCLI(DataExport())
