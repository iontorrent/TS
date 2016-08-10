#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
# RunTransfer plugin
import os
import sys
import json
import time
import glob
import traceback
import subprocess
from ion.plugin import *

class RunTransfer(IonPlugin):
    '''Main class definition for this plugin'''
    version = '5.2.0.0'
    DEBUG = False
    author = "bernard.puc@thermofisher.com"

    #Lists of file types to copy
    wells_files = {
        "1":[
            '1.wells',
        ],
    }
    sigproc_files = {
        "2":[
            'bfmask.bin',
            'bfmask.stats',
            'analysis.bfmask.bin',
            'analysis.bfmask.stats',
        ],
        "3":[
            'Bead_density_20.png',
            'Bead_density_70.png',
            'Bead_density_200.png',
            'Bead_density_1000.png',
            'Bead_density_raw.png',
            'Bead_density_contour.png',
        ],
        "4":[
            'avgNukeTrace_*.txt',
            'analysis_return_code.txt',
            'processParameters.txt',
            'MD5SUMS',
        ],
    }


    def set_upload_status(self, level, file):
        #Updates the webpage showing status of the file transfer
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print ("Could not write status report")
            print traceback.format_exc()
            raise

        display_fs.write("<html><head>\n")
        display_fs.write("<link href=\"/pluginMedia/RunTransfer/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body><center>\n")

        if level == '1':
            display_fs.write("<img src=\"/pluginMedia/%s/images/progress/wells.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/bfmask.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/nuke.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/bead.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/explog.png\">" % self.plugin_name)
        elif level == '2':
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/wells.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/progress/bfmask.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/nuke.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/bead.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/explog.png\">" % self.plugin_name)
        elif level == '3':
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/wells.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bfmask.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/progress/nuke.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/bead.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/explog.png\">" % self.plugin_name)
        elif level == '4':
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/wells.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bfmask.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/nuke.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/progress/bead.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/waiting/explog.png\">" % self.plugin_name)
        elif level == '5':
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/wells.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bfmask.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/nuke.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bead.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/progress/explog.png\">" % self.plugin_name)

        display_fs.write("<p><h2>Status:</h2><small>Uploading %s</small>\n" % file)
        if self.is_proton and self.transferred_blocks != None:
            display_fs.write("<p>%s blocks transferred</p>" % (self.transferred_blocks))
        display_fs.write("</center></body></html>\n")
        display_fs.close()

        return


    def create_upload_dir(self, location):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        cmd = "/usr/bin/sshpass -p %s ssh %s@%s \"mkdir -p %s\"" % (self.user_password,
                                                            self.user_name,
                                                            self.server_ip,
                                                            os.path.join(self.upload_path,location))
        if self.DEBUG: print "Execute: %s" % cmd
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        if p1.returncode == 0:
            print(stdout)
        else:
            raise Exception(stderr)

    def copy_files(self, sigproc_dir, upload_path):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        for k,a in self.wells_files.iteritems():
            for filename in a:
                filename = os.path.join(sigproc_dir,filename)
                if not os.path.exists(filename):
                    print ("Does Not Exist: %s" % (filename))
                else:
                    self.set_upload_status(k,filename)
                    destination_path = os.path.join(upload_path)
                    self.file_transport_rsync(filename,destination_path)

        for k,a in sorted(self.sigproc_files.iteritems(), key=lambda k: int(k[0])):
            for file in a:
                for filename in glob.glob(os.path.join(sigproc_dir,file)):
                    if not os.path.exists(filename):
                        if file.startswith('Bead_density_'):
                            filename = os.path.join(self.sigproc_dir,'Bead_density_raw.png')
                        elif 'analysis.bfmask.bin' == file:
                            filename = os.path.join(self.sigproc_dir,'bfmask.bin')
                        elif 'analysis.bfmask.stats' == file:
                            filename = os.path.join(self.sigproc_dir,'bfmask.stats')
                        destination_path = os.path.join(upload_path,file)
                    else:
                        destination_path = os.path.join(upload_path)
                        
                    if not os.path.exists(filename):
                        print ("Does Not Exist: %s" % (filename))
                    else:
                        self.set_upload_status(k,filename)
                        self.file_transport_rsync(filename,destination_path)

    def copy_explog(self):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        # for explog.txt

        def from_pgm_zip(results_dir):
            '''
            Extract explog.txt from the pgm_logs.zip file
            '''
            import zipfile
            unzipped_path = '/tmp/explog.txt'
            with zipfile.ZipFile(os.path.join(results_dir,'pgm_logs.zip'),mode='r') as zzzz:
                zzzzinfo = zzzz.getinfo('explog.txt')
                zzzz.extract(zzzzinfo,'/tmp')
            return unzipped_path

        # First look in raw data directory
        filename = os.path.join(self.raw_data_dir,'explog.txt')
        if not os.path.exists(filename):
            # Next look in parent of raw data (case:Proton data)
            filename = os.path.join(os.path.dirname(self.raw_data_dir),'explog.txt')
            if not os.path.exists(filename):
                # Next look in the report directory
                filename = os.path.join(self.analysis_dir,'explog.txt')
                if not os.path.exists(filename):
                    # Next look in the pgm_logs.zip file
                    try:
                        filename = from_pgm_zip(self.results_dir)
                    except:
                        print traceback.format_exc()
                        
        if not os.path.exists(filename):
            print ("Does Not Exist: explog.txt")
        else:
            self.set_upload_status("5",filename)    # Hardcoded status for status page
            destination_path = os.path.join(self.upload_path)
            self.file_transport_rsync(filename,destination_path)

    def transfer_plan(self):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        filename = os.path.join(self.output_dir,'plan_params.json')
        # write file with plan info
        with open(filename,'w') as f:
            json.dump(self.plan, f)
        self.file_transport_rsync(filename,self.upload_path)

    def transfer_pgm(self):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        # Create remote directory
        self.create_upload_dir("sigproc_results")
        # Copy files
        self.copy_files(self.sigproc_dir,os.path.join(self.upload_path,"sigproc_results"))
        self.copy_explog()
        self.set_file_permissions()

        return

    def transfer_fullchip(self):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        self.create_upload_dir("")
        self.transferred_blocks = 0
        self.total_blocks = 96
        work_dir = os.path.join(self.raw_data_dir,"onboard_results","sigproc_results")
        for block_dir in [block_dir for block_dir in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir,block_dir))]:
            if 'thumbnail' in block_dir:
                continue
            self.transferred_blocks += 1
            target_dir = os.path.join("onboard_results","sigproc_results",block_dir)
            self.create_upload_dir(target_dir)
            print "######\nINFO: processing %s (%d)\n######" % (block_dir,self.transferred_blocks)
            sys.stdout.flush()
            self.copy_files(os.path.join(self.raw_data_dir,target_dir),os.path.join(self.upload_path,target_dir))
        self.copy_files(os.path.join(self.raw_data_dir,"onboard_results","sigproc_results"),os.path.join(self.upload_path,"onboard_results","sigproc_results"))
        self.copy_explog()
        self.set_file_permissions()

        return


    def file_transport(self,filename,destination_path):
        #NOTE: -p option does not preserve file permissions
        cmd = 'pscp -pw %s %s %s@%s:%s' % (self.user_password,filename,self.user_name,self.server_ip,destination_path)
        if self.DEBUG: print "Execute: %s" % cmd
        sys.stdout.flush()
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        if p1.returncode == 0:
            print(stdout)
        else:
            print(stderr)
        sys.stdout.flush()
        return


    def file_transport_rsync(self, filename,destination_path):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        cmd = 'expect %s/scripts/rsync_connection.sh %s %s %s %s %s' % (self.plugin_dir,self.user_name,self.user_password,self.server_ip,filename,destination_path)
        if self.DEBUG: print "Execute: %s" % cmd
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        if p1.returncode == 0:
            print(stdout)
        else:
            print(stderr)
        sys.stdout.flush()
        return


    def set_file_permissions(self):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        launchy = "chmod -R a+w %s" % self.upload_path
        cmd = "sshpass -p %s ssh %s@%s \"cd %s; %s > stdout.log 2>&1\""% (self.user_password,
                                                                          self.user_name,
                                                                          self.server_ip,
                                                                          self.upload_path,
                                                                          launchy)
        if self.DEBUG: print "Execute: %s" % cmd
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        if p1.returncode == 0:
            print(stdout)
        else:
            print(stderr)
        return

    def start_reanalysis(self):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        self.show_standard_status("<p><h2>Status:</h2><small>Launching Analysis</small><img src=\"/site_media/jquery/colorbox/images/loading.gif\" alt=\"Running Plugin\" style=\"float:center\"></img></p>\n")
        if self.thumbnail_only == 'on':
            launchy = "python /opt/ion/iondb/bin/from_wells_analysis.py --thumbnail-only %s" % self.upload_path
        else:
            launchy = "python /opt/ion/iondb/bin/from_wells_analysis.py %s" % self.upload_path
        cmd = "sshpass -p %s ssh %s@%s \"cd %s; %s > stdout.log 2>&1\""% (self.user_password,
                                                                          self.user_name,
                                                                          self.server_ip,
                                                                          self.upload_path,
                                                                          launchy)
        if self.DEBUG: print "Execute: %s" % cmd
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        if p1.returncode == 0:
            print(stdout)
        else:
            print(stderr)

        return

    def final_status(self):
        #Not the final status of the reanalysis, but the final status of the renalysis kick-off.
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        time.sleep(70)
        stat_line = ""  # provide default
        cmd = "sshpass -p %s ssh %s@%s \"cd %s; cat stdout.log\"" % (self.user_password,
                                                                     self.user_name,
                                                                     self.server_ip,
                                                                     self.upload_path)
        if self.DEBUG: print "Execute: %s" % cmd
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        stdout, stderr = p1.communicate()
        if p1.returncode == 0:
            print(stdout)
        else:
            print(stderr)
        output = stdout

        #Parse the output for keywords
        for line in output.splitlines():
            if "STATUS: Error" in line:
                # There was an error launching re-analysis
                stat_line = "<p><h2>Status:</h2><small><font color=\"red\">Import failed with following error:</font>\n"
                for line in output.splitlines():
                    stat_line += "%s</br>" % line
                stat_line += "</small>\n"
            elif "REPORT-URL" in line:
                # REPORT-URL: <path to report>
                link = "http://"+self.server_ip+line.split(":")[1].strip()
                stat_line = "<p><h2>Status:</h2><small>Report available at <a href=\"%s\" target=\"_blank\">%s</a></small>\n" % (link,link)

        self.show_standard_status(stat_line)

    # method to display initial status view
    def show_standard_status(self, stat_line):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        #print report
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print ("Could not write status report")
            print traceback.format_exc()
            raise

        display_fs.write("<html><head>\n")
        display_fs.write("<link href=\"/pluginMedia/RunTransfer/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body>\n")
        if self.DEBUG:
            display_fs.write("plugin_name = %s</br>" % self.plugin_name)
            display_fs.write("results_dir = %s</br>" % self.results_dir)
            display_fs.write("raw_data_dir = %s</br>" % self.raw_data_dir)
            display_fs.write("results_dir_base = %s</br>" % self.results_dir_base)
            display_fs.write("output_dir = %s</br>" % self.output_dir)
            display_fs.write("platform = %s</br>" % self.platform)
            display_fs.write("server_ip = %s</br>" % self.server_ip)
            display_fs.write("user_name = %s</br>" % self.user_name)
            display_fs.write("user_password = %s</br>" % self.user_password)
            display_fs.write("upload_path = %s</br>" % self.upload_path)
            display_fs.write("thumbnail_only = %s</br>" % self.thumbnail_only)
        display_fs.write("<center>\n")
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/wells.png\">" % self.plugin_name)
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bfmask.png\">" % self.plugin_name)
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/nuke.png\">" % self.plugin_name)
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bead.png\">" % self.plugin_name)
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/explog.png\">" % self.plugin_name)    #status
        if self.is_proton and self.transferred_blocks != None:
            display_fs.write("<p>%s blocks transferred</p>" % (self.transferred_blocks))
        display_fs.write("<p> %s </p>" % stat_line)
        display_fs.write("</center></body></html>\n")
        display_fs.close()

        return

    # method to clear initial status view (if previously run, previous status is cleared)
    def init_status_page(self, stat_line):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print ("Could not write status report")
            print traceback.format_exc()
            raise

        display_fs.write("<html><head>\n")
        display_fs.write("<link href=\"/pluginMedia/RunTransfer/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body>\n")
        display_fs.write("<center>\n")
        display_fs.write("<p> %s </p>" % stat_line)
        display_fs.write("</center></body></html>\n")
        display_fs.close()

        return

    def plugin_not_configured_error(self):
        print ("Function: %s()" % sys._getframe().f_code.co_name)
        #print report
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print ("Could not write status report")
            print traceback.format_exc()
            raise

        display_fs.write("<html><head>\n")
        display_fs.write("<link href=\"/pluginMedia/RunTransfer/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body>\n")
        display_fs.write("<center>\n")
        display_fs.write("<p> %s </p>" % "PLUGIN IS NOT CONFIGURED.")
        display_fs.write("<p> %s </p>" % "Run the global configuration for this plugin from the <a href=\"/configure/plugins\" target=\"_blank\">Plugins page</a>.")
        display_fs.write("</center></body></html>\n")
        display_fs.close()



    # main method of plugin execution
    def launch(self, data=None):

        try:
            # Start-up activities (from old launch.sh)
            #clean up old keys just in case user has brought down the vm and then spun it up again and want to avoid ssh warning
            #if os.path.exists(os.path.expanduser('~/.ssh/known_hosts')):
            #    os.unlink(os.path.expanduser('~/.ssh/known_hosts'))
            #if os.path.exists(os.path.expanduser('~/.putty/sshhostkeys')):
            #    os.unlink(os.path.expanduser('~/.putty/sshhostkeys'))
            # FST-783: user ionian's ssh keys get wiped out - not good if other plugins setup ssh keys.

            # Other class variables
            self.total_blocks = None
            self.transferred_blocks = None
            self.is_proton = False

            # Gather variables
            with open('startplugin.json', 'r') as fh:
                spj = json.load(fh)
                self.results_dir     = spj['runinfo']['report_root_dir']
                self.raw_data_dir    = spj['runinfo']['raw_data_dir']
                self.results_dir_base = os.path.basename(spj['runinfo']['report_root_dir'])
                self.output_dir      = spj['runinfo']['results_dir']
                self.plugin_name     = spj['runinfo']['plugin_name']
                self.platform        = spj['runinfo']['platform']
                self.sigproc_dir     = spj['runinfo']['sigproc_dir']
                self.analysis_dir    = spj['runinfo']['analysis_dir']
                self.plugin_dir      = spj['runinfo']['plugin_dir']
                try:
                    self.server_ip       = spj['pluginconfig']['ip']
                    self.user_name       = spj['pluginconfig']['user_name']
                    self.user_password   = spj['pluginconfig']['user_password']
                    upload_path_local    = spj['pluginconfig']['upload_path']
                    if self.server_ip == "" or self.user_name == "" or self.user_password == "" or upload_path_local == "":
                        raise Exception()
                except:
                    # If these fail, then plugin is not configured
                    self.plugin_not_configured_error()
                    return True

                self.upload_path     = os.path.join(spj['pluginconfig']['upload_path'],self.results_dir_base+'_foreign')
                self.thumbnail_only  = spj['pluginconfig'].get('thumbnailonly', 'off')
                self.plan            = spj.get('plan','')

            # Initialize output status page
            self.init_status_page("Hold on...")

            print("=====\nParameters used in this plugin:\n")
            print("plugin_name = %s" % self.plugin_name)
            print("results_dir = %s" % self.results_dir)
            print("raw_data_dir = %s" % self.raw_data_dir)
            print("results_dir_base = %s" % self.results_dir_base)
            print("output_dir = %s" % self.output_dir)
            print("platform = %s" % self.platform)
            print("server_ip = %s" % self.server_ip)
            print("user_name = %s" % self.user_name)
            print("user_password = %s" % self.user_password)
            print("upload_path = %s" % self.upload_path)
            print("thumbnail_only = %s" % self.thumbnail_only)
            print("")

            #create connection to generated ssh entry
            cmd = "expect %s/scripts/ssh_connection.sh %s %s " % (self.plugin_dir,self.user_name,self.server_ip)
            if self.DEBUG: print "Execute: %s" % cmd
            p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
            stdout, stderr = p1.communicate()
            if p1.returncode == 0:
                print(stdout)
            else:
                raise Exception(stderr)

            cmd = "expect %s/scripts/pscp_connection.sh %s %s %s > %s/temp.txt" % (self.plugin_dir,self.user_password,self.user_name,self.server_ip,self.output_dir)
            if self.DEBUG: print "Execute: %s" % cmd
            p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
            stdout, stderr = p1.communicate()
            if p1.returncode == 0:
                print(stdout)
            else:
                raise Exception(stderr)

            # Display initial status html
            self.show_standard_status("Starting")

            # Determine Dataset Type
            if self.platform.lower() in ['proton', 's5']:
                self.is_proton = True
            else:
                self.is_proton = False

            if self.is_proton:
                if self.thumbnail_only == 'on':
                    # Get thumbnail data
                    self.transfer_pgm()
                else:
                    # Get fullchip data
                    if 'thumbnail' in os.path.basename(self.raw_data_dir):
                        self.raw_data_dir = os.path.dirname(self.raw_data_dir)
                    self.transfer_fullchip()
            else:
                # Get PGM data
                self.thumbnail_only = 'off'
                self.transfer_pgm()

            # Save extra parameters from Plan
            self.transfer_plan()

            sys.stdout.flush()

            # Start the re-analysis on the target server
            self.start_reanalysis()

            # Collect the final status
            self.final_status()
        except:
            print traceback.format_exc()
            self.show_standard_status(traceback.format_exc())

        return True


    def report(self):
        pass

# dev use only - makes testing easier
if __name__ == "__main__": PluginCLI(RunTransfer())
