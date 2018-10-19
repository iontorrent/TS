#!/usr/bin/env python
# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
# RunTransfer plugin
import os
import json
import logging
import glob
import requests
import traceback
import zipfile
from distutils.version import LooseVersion
from ion.plugin import *
from ion.utils.explogparser import parse_log
from ftplib import FTP, error_perm
from django.utils.functional import cached_property

DJANGO_FTP_PORT = 8021
PLAN_PARAMS_FILENAME = 'plan_params.json'
CHEF_SUMMARY_FILENAME = 'chef_params.json'
EXPLOG_FILENAME = 'explog.txt'

# the list of required files for the root files
PGMSTYLE_REQUIRED_FILES = ['1.wells', 'analysis.bfmask.bin', 'processParameters.txt', 'avgNukeTrace_ATCG.txt', 'avgNukeTrace_TCAG.txt', 'bfmask.stats', 'bfmask.bin', 'analysis.bfmask.stats', 'analysis_return_code.txt', 'sigproc.log']
BLOCKSTYLE_SIGPROC_ROOT_LEVEL_REQUIRED_FILES = ['avgNukeTrace_ATCG.txt', 'avgNukeTrace_TCAG.txt', 'analysis.bfmask.stats']
REQUIRED_RESULTS_FILES = [PLAN_PARAMS_FILENAME]
OPTIONAL_RESULTS_FILES = [CHEF_SUMMARY_FILENAME]
OPTIONAL_SIGNAL_FILE_PATTERNS = ['Bead_density_20.png', 'Bead_density_70.png', 'Bead_density_200.png', 'Bead_density_1000.png', 'Bead_density_raw.png', 'Bead_density_contour.png']


class RunTransfer(IonPlugin):
    """Main class definition for this plugin"""

    version = '5.10.0.1'
    author = "bernard.puc@thermofisher.com"
    runtypes = [RunType.FULLCHIP, RunType.COMPOSITE]

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
    is_proton = False
    total_blocks = None
    transferred_blocks = None
    json_head = {'Content-Type': 'application/json'}
    rest_auth = None
    spj = dict()

    @cached_property
    def ftp_client(self):
        """Helper property to get an ftp client"""
        client = FTP()
        client.connect(host=self.server_ip, port=DJANGO_FTP_PORT)
        client.login(user=self.user_name, passwd=self.user_password)
        return client

    @cached_property
    def barcodedata(self):
        """Gets the barcodes.json data"""
        with open('barcodes.json', 'r') as handle:
            return json.load(handle)

    def get_list_of_files_common(self):
        """Gets a list of common files to transfer"""
        plugin_results_dir = self.spj['runinfo']['plugin']['results_dir']

        file_transfer_list = list()
        for required_file in REQUIRED_RESULTS_FILES:
            file_transfer_list.append((os.path.join(plugin_results_dir, required_file), self.upload_path))

        for optional_file in OPTIONAL_RESULTS_FILES:
            optional_path = os.path.join(plugin_results_dir, optional_file)
            if os.path.exists(optional_path):
                file_transfer_list.append((optional_path, self.upload_path))

        return file_transfer_list

    def get_list_of_files_pgmstyle(self, root_sigproc_dir):
        """This helper method will get a list of the tuples (source path, destination path) and verify that the required files are present for the pgm style"""
        file_transfer_list = list()

        # because the explog can be in multiple locations we are going to have to get it from there
        file_transfer_list.append((self.setup_explog(), self.upload_path))

        # verify the files before we attempt to copy them
        for filename in PGMSTYLE_REQUIRED_FILES:
            filename = os.path.join(root_sigproc_dir, filename)
            if not os.path.exists(filename):
                raise Exception("The required file %s does not exists and thus the run transfer cannot be completed." % filename)
            file_transfer_list.append((filename, os.path.join(self.upload_path, "onboard_results", "sigproc_results")))
        return file_transfer_list

    def get_list_of_files_blockstyle(self, root_sigproc_dir, block_dirs):
        """This helper method will get a list of the tuples (source path, destination path) and verify that the required files are present for the block style"""
        file_transfer_list = list()

        # because the explog can be in multiple locations we are going to have to get it from there
        file_transfer_list.append((self.setup_explog(), self.upload_path))

        # iterate through all of the signal processing output directories
        dst_sigproc_dir = os.path.join(self.upload_path, "onboard_results", "sigproc_results")
        # now collect required files from the root of the results directory
        for filename in BLOCKSTYLE_SIGPROC_ROOT_LEVEL_REQUIRED_FILES:
            file_transfer_list.append((os.path.join(root_sigproc_dir, filename), dst_sigproc_dir))

        for block_dir in block_dirs:
            destination_directory = os.path.join(dst_sigproc_dir, os.path.basename(block_dir))

            # verify the files before we attempt to copy them
            for filename in PGMSTYLE_REQUIRED_FILES:
                filename = os.path.join(block_dir, filename)
                if not os.path.exists(filename):
                    raise Exception("The required file %s does not exists and thus the run transfer cannot be completed." % filename)
                file_transfer_list.append((filename, destination_directory))

        return file_transfer_list

    def copy_files(self, file_transfer_list):
        """This helper method will copy over all of the files in the directory"""
        # assuming we have all of the required files on the local system, we will now do the transfer

        destination_directories = set([destination for _, destination in file_transfer_list])
        for destination_directory in destination_directories:
            self.create_remote_directory(destination_directory)

        total_transferred = 0
        total_files = len(file_transfer_list)
        for source_file_path, destination_directory in file_transfer_list:
            self.set_upload_status(total_transferred, total_files, source_file_path)
            self.file_transport(source_file_path, destination_directory)
            total_transferred += 1

    def setup_explog(self):
        """This method will find the experiment log and return it's location"""
        # First look in raw data directory
        original = os.path.join(self.raw_data_dir, EXPLOG_FILENAME)
        if not os.path.exists(original):
            # Next look in parent of raw data (case:Proton data)
            original = os.path.join(os.path.dirname(self.raw_data_dir), EXPLOG_FILENAME)

        # Next look in the report directory
        if not os.path.exists(original):
            original = os.path.join(self.analysis_dir, EXPLOG_FILENAME)

        # Next look in the pgm_logs.zip file
        if not os.path.exists(original) and os.path.exists(os.path.join(self.results_dir, 'pgm_logs.zip')):
            original = os.path.join(self.raw_data_dir, EXPLOG_FILENAME)
            with zipfile.ZipFile(os.path.join(self.results_dir, 'pgm_logs.zip'), mode='r') as pgm_zip_hangle:
                explog_info = pgm_zip_hangle.getinfo(EXPLOG_FILENAME)
                pgm_zip_hangle.extract(explog_info, self.raw_data_dir)

        # read in the exp log
        with open(original, 'r') as original_handle:
            explog = parse_log(original_handle.read())

        # HACK ALERT!  In order to make sure we don't go over the maximum length of the experiment name (currently 128 characters) by the
        # appending of the _foreign string in the from_wells_analysis.py logic, we are going to have to add a check to see if it can fit
        # into the data base constraints with the appending of the foreign string
        if len(explog['experiment_name'] + "_foreign") > 128:
            raise Exception("We cannot transfer this result due to the length of the experiment name.")

        return original

    def create_remote_directory(self, directory_path):
        """Helper method to create the directory on the remote server via ftp"""
        directories = filter(None, directory_path.split('/'))
        cur_dir = '/'
        for sub_directory in directories:
            # Create remote directory
            cur_dir = os.path.join(cur_dir, sub_directory)
            try:
                self.ftp_client.mkd(cur_dir)
            except error_perm:
                pass

    def file_transport(self, filename, destination_path):
        """Transfers a file across the ftp"""
        # delete the old file
        try:
            self.ftp_client.delete(os.path.join(destination_path, os.path.basename(filename)))
        except:
            # don't do anything in case this fails....
            pass

        # push the new file
        try:
            self.ftp_client.cwd(destination_path)
            self.ftp_client.storbinary('STOR ' + os.path.basename(filename), open(filename, 'rb'))
        except error_perm as exc:
            if '550' in exc.message:

                print(traceback.format_exc())
                print("550 Error while attempting to transfer file %s to %s" % (filename, destination_path))
                print(filename + " -> " + destination_path)
                raise Exception("The destination already contains the files and cannot overwrite them.  This is most likely due to a previous execution of Run Transfer.")
            else:
                raise

    def start_reanalysis(self):
        """Set the status for a reanalysis"""
        self.show_standard_status("<p><h2>Status:</h2><small>Launching Analysis</small><img src=\"/site_media/jquery/colorbox/images/loading.gif\" alt=\"Running Plugin\" style=\"float:center\"></img></p>\n")
        analysis_params = {'directory': self.upload_path, 'is_thumbnail': False}
        response = requests.post('http://' + self.server_ip + '/rundb/api/v1/experiment/from_wells_analysis/', data=json.dumps(analysis_params), headers=self.json_head, auth=self.rest_auth)
        response.raise_for_status()
        return response.content

    def show_standard_status(self, stat_line):
        """method to display initial status view"""

        # replace new lines with line breaks
        stat_line = stat_line.replace("\n", "<br />")
        with open('status_block.html', "wb") as display_fs:
            display_fs.write("<html><head>\n")
            display_fs.write("<link href=\"/pluginMedia/RunTransfer/bootstrap.min.css\" rel=\"stylesheet\">\n")
            display_fs.write("</head><body>\n")
            display_fs.write("<center>\n")
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/wells.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bfmask.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/nuke.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/bead.png\">" % self.plugin_name)
            display_fs.write("<img src=\"/pluginMedia/%s/images/complete/explog.png\">" % self.plugin_name)
            if self.is_proton and self.transferred_blocks:
                display_fs.write("<p>%s blocks transferred</p>" % self.transferred_blocks)
            display_fs.write("<p> %s </p>" % stat_line)
            display_fs.write("</center></body></html>\n")

    def set_upload_status(self, total_transfered, total_files, file_uploading):
        """Creates the update page"""
        # Updates the webpage showing status of the file transfer
        progress = int(float(total_transfered) / float(total_files) * 100)

        with open('status_block.html', "wb") as display_fs:
            display_fs.write("<html><head>\n")
            display_fs.write("<link href=\"/site_media/resources/bootstrap/css/bootstrap.min.css\" rel=\"stylesheet\">\n")
            display_fs.write("<meta http-equiv=\"refresh\" content=\"10\" >")
            display_fs.write("</head><body><center>\n")
            display_fs.write("<p><h2>Status:</h2><small>Uploading %s</small>\n" % file_uploading)
            display_fs.write("<div class=\"progress\"><div class=\"bar\" style=\"width: %d%%;\"></div></div>" % progress)
            display_fs.write("</center></body></html>\n")

    def init_status_page(self, stat_line):
        """method to clear initial status view (if previously run, previous status is cleared)"""
        stat_line = stat_line.replace("\n", "<br />")
        with open('status_block.html', "wb") as display_fs:
            display_fs.write("<html><head>\n")
            display_fs.write("<link href=\"/site_media/resources/bootstrap/css/bootstrap.min.css\" rel=\"stylesheet\">\n")
            display_fs.write("</head><body>\n")
            display_fs.write("<center>\n")
            display_fs.write("<p> %s </p>" % stat_line)
            display_fs.write("</center></body></html>\n")

    def plugin_not_configured_error(self):
        """Write out the status that the plugin is not configured"""
        with open('status_block.html', "wb") as display_fs:
            display_fs.write("<html><head>\n")
            display_fs.write("<link href=\"/site_media/resources/bootstrap/css/bootstrap.min.css\" rel=\"stylesheet\">\n")
            display_fs.write("</head><body>\n")
            display_fs.write("<center>\n")
            display_fs.write("<p> PLUGIN IS NOT CONFIGURED. </p>")
            display_fs.write("<p> Run the global configuration for this plugin from the <a href=\"/configure/plugins\" target=\"_blank\">Plugins page</a>. </p>")
            display_fs.write("</center></body></html>\n")

    def check_version(self):
        """This method will check the version of the remote site to make sure that it's a compatible version"""
        # test to make sure both of them must have identical versions
        api_version_directory = '/rundb/api/v1/torrentsuite/version'
        local_version_response = requests.get(self.spj['runinfo']['net_location'] + api_version_directory)
        local_version_response.raise_for_status()
        local_version = '5.10.0.0'

        remote_version_response = requests.get('http://' + self.server_ip + api_version_directory)
        remote_version_response.raise_for_status()
        remote_version = '5.10.0.0'

        if not remote_version:
            raise Exception('Could not establish version of remote computer, exiting.')

        if local_version != remote_version:
            raise Exception("In order to transfer runs the remote torrent suite must have the identical version.")

        if LooseVersion(remote_version) < LooseVersion('5.3.0.0'):
            raise Exception('The remote server\'s version of Torrent Suite is not compatible.')

    def check_for_localhost(self):
        """This method will make sure that we are not trying to transfer to ourselves"""
        system_id_api_endpoint = "/rundb/api/v1/ionmeshnode/system_id/"
        remote_system_id_response = requests.get("http://" + self.server_ip + system_id_api_endpoint, auth=self.rest_auth)
        remote_system_id_response.raise_for_status()
        remote_system_id = json.loads(remote_system_id_response.content)['system_id']

        api_key_args = {'api_key': self.spj['runinfo']['api_key'], 'pluginresult': str(self.spj['runinfo']['pluginresult'])}
        local_system_id_response = requests.get(self.spj['runinfo']['net_location'] + system_id_api_endpoint, params=api_key_args)
        local_system_id_response.raise_for_status()
        local_system_id = json.loads(local_system_id_response.content)['system_id']

        if local_system_id == remote_system_id:
            raise Exception("The remote system is the same and this one.  Transferring to the same machine is not allowed.")

    def launch(self, data=None):
        """main method of plugin execution"""

        def find_reference_in_list(short_name, reference_genomes_list):
            """Helper method to that detects if the short name is in the list"""
            for reference_genome_item in reference_genomes_list:
                if reference_genome_item['short_name'] == short_name:
                    return True
            return False

        try:
            # turn off the logging to prevent logging of the url's with the api keys
            logging.getLogger("requests").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)

            # Gather variables
            self.spj = self.startplugin
            self.results_dir = self.spj['runinfo']['report_root_dir']
            self.raw_data_dir = self.spj['runinfo']['raw_data_dir']
            self.results_dir_base = os.path.basename(self.spj['runinfo']['report_root_dir'])
            self.output_dir = self.spj['runinfo']['results_dir']
            self.plugin_name = self.spj['runinfo']['plugin_name']
            self.sigproc_dir = self.spj['runinfo']['sigproc_dir']
            self.analysis_dir = self.spj['runinfo']['analysis_dir']
            self.plugin_dir = self.spj['runinfo']['plugin_dir']
            api_key = self.spj['runinfo']['api_key']
            try:
                self.server_ip = self.spj['pluginconfig']['ip']
                self.user_name = self.spj['pluginconfig']['user_name']
                upload_path_local = self.spj['pluginconfig']['upload_path']
                if self.server_ip == "" or self.user_name == "" or upload_path_local == "":
                    raise Exception()
            except:
                # If these fail, then plugin is not configured
                self.plugin_not_configured_error()
                return True

            self.upload_path = os.path.join(self.spj['pluginconfig']['upload_path'], self.results_dir_base + '_foreign')
            # Determine Dataset Type
            self.is_proton = self.spj['runinfo']['platform'].lower() in ['proton', 's5']

            if self.spj['runplugin']['run_type'].lower() == 'thumbnail':
                self.show_standard_status("The plugin is set to only transfer non-thumbnail data.")
                return True

            plan = self.spj.get('plan', dict())
            chef_summary = self.spj.get('chefSummary', dict())

            # this method will check version compatibility
            self.check_version()

            # attempt to retrieve the password from secure storage
            secret_args = {
                'name': 'RunTransferConfig-' + self.server_ip + "-" + self.user_name,
                'api_key': api_key,
                'pluginresult': str(self.spj['runinfo']['pluginresult']),
            }
            secret_response = requests.get(self.spj['runinfo']['net_location'] + '/security/api/v1/securestring/', params=secret_args)
            secret_response.raise_for_status()
            json_secret = json.loads(secret_response.content)

            if 'objects' not in json_secret or len(json_secret['objects']) == 0:
                raise Exception("Could not get password from secure storage.")
            self.user_password = json_secret['objects'][0]['decrypted']
            self.rest_auth = (self.user_name, self.user_password)

            # check to make sure that we are not attempting to transfer to the exact same machine
            self.check_for_localhost()

            # append the TS source system id to the plan which we will get from the global config which there should always be one and only one
            global_config_response = requests.get(self.spj['runinfo']['net_location'] + '/rundb/api/v1/globalconfig/', params={'api_key': api_key, 'pluginresult': str(self.spj['runinfo']['pluginresult'])})
            global_config_response.raise_for_status()
            global_config = json.loads(global_config_response.content)
            plan['runTransferFromSource'] = global_config['objects'][0]['site_name']

            # check that all of the references are available on the remote server
            reference_request = requests.get('http://' + self.server_ip + '/rundb/api/v1/referencegenome/?enabled=true', auth=self.rest_auth)
            reference_request.raise_for_status()
            reference_genomes = json.loads(reference_request.content)['objects']
            for barcode_name, barcode_data in self.barcodedata.items():
                reference_short_name = barcode_data.get('reference', '')
                if barcode_name == 'nomatch' or not reference_short_name:
                    continue

                if not find_reference_in_list(reference_short_name, reference_genomes):
                    raise Exception("The remote execution will not be run because the remote site does not have the reference " + reference_short_name)

            # Display initial status html
            self.show_standard_status("Starting")

            # prepare transient files
            if plan:
                with open(os.path.join(self.output_dir, PLAN_PARAMS_FILENAME), 'w') as plan_file_handle:
                    json.dump(plan, plan_file_handle)

            if chef_summary:
                with open(os.path.join(self.output_dir, CHEF_SUMMARY_FILENAME), 'w') as chef_file_handle:
                    json.dump(chef_summary, chef_file_handle)

            # get a list of all of the files which will be transferred
            file_transfer_list = list()
            src_sigproc_dir = os.path.join(self.results_dir, 'sigproc_results')
            if self.is_proton:
                # generate a list of all of the block directories
                block_directories = [os.path.join(src_sigproc_dir, block_dir) for block_dir in os.listdir(src_sigproc_dir) if os.path.isdir(os.path.join(src_sigproc_dir, block_dir)) and 'thumbnail' not in block_dir]

                # first collect a list of all of the files to transfer from all of the block directories
                file_transfer_list = self.get_list_of_files_blockstyle(src_sigproc_dir, block_directories)
            else:
                # first collect a list of all of the files to transfer from all of the block directories
                file_transfer_list = self.get_list_of_files_pgmstyle(src_sigproc_dir)

            file_transfer_list += self.get_list_of_files_common()
            # now transfer the files across the transport layer
            # for file_pair in file_transfer_list:
            #     print(file_pair[0] + "-->" + file_pair[1] + "\n")
            self.copy_files(file_transfer_list)

            # Start the re-analysis on the target server
            self.show_standard_status(self.start_reanalysis())
            return True

        except requests.exceptions.ConnectionError as exc:
            print(exc)
            self.show_standard_status("<strong>Could not create a connection to the server %s.</strong><br />" % self.server_ip)
            raise

        except Exception as exc:
            print(exc)
            self.show_standard_status("<strong>There was an issue running the plugin</strong><br />" + str(exc))
            raise


# dev use only - makes testing easier
if __name__ == "__main__":
    PluginCLI(RunTransfer())
