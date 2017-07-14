#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
# RunTransfer plugin
import os
import sys
import json
import glob
import requests
import traceback
from distutils.version import LooseVersion
from ion.plugin import *
from ftplib import FTP, error_perm
from django.utils.functional import cached_property

DJANGO_FTP_PORT = 8021


class RunTransfer(IonPlugin):
    """Main class definition for this plugin"""

    version = '5.4.0.7'
    author = "bernard.puc@thermofisher.com"

    results_dir = None
    raw_data_dir = None
    results_dir_base = None
    output_dir = None
    plugin_name = None
    sigproc_dir = None
    analysis_dir = None
    plugin_dir = None
    server_ip = None
    user_name = None
    user_password = None
    upload_path = None
    thumbnail_only = None
    port = None
    plan = None
    chefSummary = None
    is_proton = False
    total_blocks = None
    transferred_blocks = None
    json_head = {'Content-Type': 'application/json'}
    rest_auth = None

    # Lists of file types to copy
    wells_files = {
        "1": [
            '1.wells',
        ],
    }
    sigproc_files = {
        "2": [
            'bfmask.bin',
            'bfmask.stats',
            'analysis.bfmask.bin',
            'analysis.bfmask.stats',
        ],
        "3": [
            'Bead_density_20.png',
            'Bead_density_70.png',
            'Bead_density_200.png',
            'Bead_density_1000.png',
            'Bead_density_raw.png',
            'Bead_density_contour.png',
        ],
        "4": [
            'avgNukeTrace_*.txt',
            'analysis_return_code.txt',
            'processParameters.txt',
            'MD5SUMS',
        ],
    }

    @cached_property
    def ftp_client(self):
        """Helper property to get an ftp client"""
        client = FTP()
        client.connect(host=self.server_ip, port=DJANGO_FTP_PORT)
        client.login(user=self.user_name, passwd=self.user_password)
        return client

    def set_upload_status(self, level, file):
        #Updates the webpage showing status of the file transfer
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print("Could not write status report")
            print(traceback.format_exc())
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
            display_fs.write("<p>%s blocks transferred</p>" % self.transferred_blocks)
        display_fs.write("</center></body></html>\n")
        display_fs.close()

    def copy_files(self, sigproc_dir, upload_path):

        # verify the files before we attempt to copy them
        for k, a in self.wells_files.iteritems():
            for filename in a:
                filename = os.path.join(sigproc_dir, filename)
                if not os.path.exists(filename):
                    raise Exception("Does Not Exist: %s" % filename)

        for k, a in sorted(self.sigproc_files.iteritems(), key=lambda k: int(k[0])):
            for sigproc_file in a:
                for filename in glob.glob(os.path.join(sigproc_dir, sigproc_file)):
                    if not os.path.exists(filename):
                        if sigproc_file.startswith('Bead_density_'):
                            filename = os.path.join(self.sigproc_dir, 'Bead_density_raw.png')
                        elif 'analysis.bfmask.bin' == sigproc_file:
                            filename = os.path.join(self.sigproc_dir, 'bfmask.bin')
                        elif 'analysis.bfmask.stats' == sigproc_file:
                            filename = os.path.join(self.sigproc_dir, 'bfmask.stats')

                    if not os.path.exists(filename):
                        raise Exception("Does Not Exist: %s" % filename)

        for k, a in self.wells_files.iteritems():
            for filename in a:
                filename = os.path.join(sigproc_dir,filename)
                self.set_upload_status(k,filename)
                destination_path = os.path.join(upload_path)
                self.file_transport(filename,destination_path)

        for k,a in sorted(self.sigproc_files.iteritems(), key=lambda k: int(k[0])):
            for sigproc_file in a:
                for filename in glob.glob(os.path.join(sigproc_dir,sigproc_file)):
                    if not os.path.exists(filename):
                        if sigproc_file.startswith('Bead_density_'):
                            filename = os.path.join(self.sigproc_dir,'Bead_density_raw.png')
                        elif 'analysis.bfmask.bin' == sigproc_file:
                            filename = os.path.join(self.sigproc_dir,'bfmask.bin')
                        elif 'analysis.bfmask.stats' == sigproc_file:
                            filename = os.path.join(self.sigproc_dir,'bfmask.stats')
                        destination_path = os.path.join(upload_path,sigproc_file)
                    else:
                        destination_path = os.path.join(upload_path)

                    if not os.path.exists(filename):
                        print ("Does Not Exist: %s" % filename)
                    else:
                        self.set_upload_status(k,filename)
                        self.file_transport(filename,destination_path)

    def copy_explog(self):
        # for explog.txt
        def from_pgm_zip(results_dir):
            """Extract explog.txt from the pgm_logs.zip file"""
            import zipfile
            unzipped_path = '/tmp/explog.txt'
            with zipfile.ZipFile(os.path.join(results_dir,'pgm_logs.zip'),mode='r') as zzzz:
                zzzzinfo = zzzz.getinfo('explog.txt')
                zzzz.extract(zzzzinfo,'/tmp')
            return unzipped_path

        # First look in raw data directory
        filename = os.path.join(self.raw_data_dir, 'explog.txt')
        if not os.path.exists(filename):
            # Next look in parent of raw data (case:Proton data)
            filename = os.path.join(os.path.dirname(self.raw_data_dir), 'explog.txt')
            if not os.path.exists(filename):
                # Next look in the report directory
                filename = os.path.join(self.analysis_dir, 'explog.txt')
                if not os.path.exists(filename):
                    # Next look in the pgm_logs.zip file
                    try:
                        filename = from_pgm_zip(self.results_dir)
                    except:
                        print(traceback.format_exc())

        if not os.path.exists(filename):
            print ("Does Not Exist: explog.txt")
        else:
            # Hardcoded status for status page
            self.set_upload_status("5", filename)
            destination_path = os.path.join(self.upload_path)
            self.file_transport(filename, destination_path)

    def transfer_plan(self):
        filename = os.path.join(self.output_dir, 'plan_params.json')
        # write file with plan info
        with open(filename,'w') as f:
            json.dump(self.plan, f)
        self.file_transport(filename, self.upload_path)

    def transfer_chef_summary(self):
        filename = os.path.join(self.output_dir, 'chef_params.json')

        # write file with chef summary info
        with open(filename,'w') as f:
            json.dump(self.chefSummary, f)
        self.file_transport(filename, self.upload_path)

    def transfer_pgm(self):
        directories = filter(None, self.upload_path.split('/'))
        directories.append("sigproc_results")
        cur_dir = '/'
        for dir in directories:
            # Create remote directory
            cur_dir = os.path.join(cur_dir, dir)
            try:
                self.ftp_client.mkd(cur_dir)
            except error_perm:
                pass

        # Copy files
        self.copy_files(self.sigproc_dir,os.path.join(self.upload_path, "sigproc_results"))
        self.copy_explog()

    def transfer_fullchip(self):

        # create the remote directory
        self.ftp_client.mkd(os.path.join(self.upload_path))

        self.transferred_blocks = 0
        self.total_blocks = 96

        # we need to create onboard results and sigproc
        self.ftp_client.mkd(os.path.join(self.upload_path, "onboard_results"))
        self.ftp_client.mkd(os.path.join(self.upload_path, "onboard_results", "sigproc_results"))
        src_sigproc_dir = os.path.join(self.raw_data_dir, "onboard_results", "sigproc_results")

        for block_dir in [block_dir for block_dir in os.listdir(src_sigproc_dir) if os.path.isdir(os.path.join(src_sigproc_dir, block_dir))]:
            if 'thumbnail' in block_dir:
                continue
            self.transferred_blocks += 1
            target_dir = os.path.join("onboard_results","sigproc_results",block_dir)

            # create the target directory on the remote machine
            self.ftp_client.mkd(os.path.join(self.upload_path, target_dir))
            print("######\nINFO: processing %s (%d)\n######" % (block_dir,self.transferred_blocks))
            sys.stdout.flush()
            self.copy_files(os.path.join(self.raw_data_dir, target_dir), os.path.join(self.upload_path, target_dir))
        self.copy_explog()

    def file_transport(self, filename, destination_path):
        """Transfers a file across the ftp"""
        self.ftp_client.cwd(destination_path)
        self.ftp_client.storbinary('STOR ' + os.path.basename(filename), open(filename, 'rb'))

    def start_reanalysis(self):
        self.show_standard_status("<p><h2>Status:</h2><small>Launching Analysis</small><img src=\"/site_media/jquery/colorbox/images/loading.gif\" alt=\"Running Plugin\" style=\"float:center\"></img></p>\n")
        analysis_params = {
            'directory': self.upload_path,
            'thumbnail_only': self.thumbnail_only
        }
        response = requests.post('http://' + self.server_ip + '/rundb/api/v1/experiment/from_wells_analysis/', data=json.dumps(analysis_params), headers=self.json_head, auth=self.rest_auth)
        if not response.ok:
            response.raise_for_status()
        return response.content

    def show_standard_status(self, stat_line):
        """method to display initial status view"""
        #print report
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print("Could not write status report")
            print(traceback.format_exc())
            raise

        # replace new lines with line breaks
        stat_line = stat_line.replace("\n", "<br />")

        display_fs.write("<html><head>\n")
        display_fs.write("<link href=\"/pluginMedia/RunTransfer/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body>\n")
        display_fs.write("<center>\n")
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/wells.png\">" % self.plugin_name)
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bfmask.png\">" % self.plugin_name)
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/nuke.png\">" % self.plugin_name)
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bead.png\">" % self.plugin_name)
        display_fs.write("<img src=\"/pluginMedia/%s/images/complete/explog.png\">" % self.plugin_name)    #status
        if self.is_proton and self.transferred_blocks:
            display_fs.write("<p>%s blocks transferred</p>" % self.transferred_blocks)
        display_fs.write("<p> %s </p>" % stat_line)
        display_fs.write("</center></body></html>\n")
        display_fs.close()

        return

    def init_status_page(self, stat_line):
        """method to clear initial status view (if previously run, previous status is cleared)"""
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print("Could not write status report")
            print(traceback.format_exc())
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
        #print report
        stat_fs_path = os.path.join(self.output_dir, 'status_block.html')
        try:
            display_fs = open(stat_fs_path, "wb")
        except:
            print ("Could not write status report")
            print(traceback.format_exc())
            raise

        display_fs.write("<html><head>\n")
        display_fs.write("<link href=\"/pluginMedia/RunTransfer/bootstrap.min.css\" rel=\"stylesheet\">\n")
        display_fs.write("</head><body>\n")
        display_fs.write("<center>\n")
        display_fs.write("<p> %s </p>" % "PLUGIN IS NOT CONFIGURED.")
        display_fs.write("<p> %s </p>" % "Run the global configuration for this plugin from the <a href=\"/configure/plugins\" target=\"_blank\">Plugins page</a>.")
        display_fs.write("</center></body></html>\n")
        display_fs.close()

    def check_version(self):
        """This method will check the version of the remote site to make sure that it's a compatible version"""
        response = requests.get('http://' + self.server_ip + '/rundb/api/v1/torrentsuite/version/', auth=self.rest_auth)
        response.raise_for_status()
        meta_version = response.json().get('meta_version', None)

        if not meta_version:
            raise Exception('Could not establish version of remote computer, exiting.')

        if LooseVersion(meta_version) < LooseVersion('5.3.0.0'):
            raise Exception('The remote server\'s version of Torrent Suite is not compatible.')

    def launch(self, data=None):
        """main method of plugin execution"""
        try:
            # Gather variables
            with open('startplugin.json', 'r') as fh:
                spj = json.load(fh)
                self.results_dir     = spj['runinfo']['report_root_dir']
                self.raw_data_dir    = spj['runinfo']['raw_data_dir']
                self.results_dir_base = os.path.basename(spj['runinfo']['report_root_dir'])
                self.output_dir      = spj['runinfo']['results_dir']
                self.plugin_name     = spj['runinfo']['plugin_name']
                self.sigproc_dir     = spj['runinfo']['sigproc_dir']
                self.analysis_dir    = spj['runinfo']['analysis_dir']
                self.plugin_dir      = spj['runinfo']['plugin_dir']
                api_key = spj['runinfo']['api_key']
                try:
                    self.server_ip       = spj['pluginconfig']['ip']
                    self.user_name       = spj['pluginconfig']['user_name']
                    upload_path_local    = spj['pluginconfig']['upload_path']
                    if self.server_ip == "" or self.user_name == "" or upload_path_local == "":
                        raise Exception()
                except:
                    # If these fail, then plugin is not configured
                    self.plugin_not_configured_error()
                    return True

                self.upload_path = os.path.join(spj['pluginconfig']['upload_path'], self.results_dir_base + '_foreign')
                self.thumbnail_only = spj['pluginconfig'].get('thumbnailonly', 'off').lower() in ['true', 'on']
                self.plan = spj.get('plan', '')
                self.chefSummary = spj.get('chefSummary', '')

            # attempt to retrieve the password from secure storage
            secret_response = requests.get('http://localhost/security/api/v1/securestring/?name=RunTransferConfig-' + self.server_ip + "-" + self.user_name + '&api_key=' + api_key)
            secret_response.raise_for_status()
            json_secret = json.loads(secret_response.content)

            if 'objects' not in json_secret or len(json_secret['objects']) == 0:
                raise Exception("Could not get password from secure storage.")
            self.user_password = json_secret['objects'][0]['decrypted']
            self.rest_auth = (self.user_name, self.user_password)

            # this method will check version compatibility
            self.check_version()

            # Display initial status html
            self.show_standard_status("Starting")

            # Determine Dataset Type
            self.is_proton = spj['runinfo']['platform'].lower() in ['proton', 's5']

            if self.is_proton:
                if self.thumbnail_only:
                    # Get thumbnail data
                    self.transfer_pgm()
                else:
                    # Get fullchip data
                    if 'thumbnail' in os.path.basename(self.raw_data_dir):
                        self.raw_data_dir = os.path.dirname(self.raw_data_dir)
                    self.transfer_fullchip()
            else:
                # Get PGM data
                self.thumbnail_only = False
                self.transfer_pgm()

            # Save extra parameters from Plan
            self.transfer_plan()

            if self.chefSummary:
                self.transfer_chef_summary()

            sys.stdout.flush()

            # Start the re-analysis on the target server
            self.show_standard_status(self.start_reanalysis())
            return True

        except requests.exceptions.ConnectionError as exc:
            print(exc)
            self.show_standard_status("<strong>Could not create a connection to the server %s.</strong><br />" % self.server_ip)
            raise

        except Exception as exc:
            print(exc)
            self.show_standard_status("<strong>There was issue problem running the plugin</strong><br />" + str(exc))
            raise


    def report(self):
        pass


# dev use only - makes testing easier
if __name__ == "__main__": PluginCLI(RunTransfer())
