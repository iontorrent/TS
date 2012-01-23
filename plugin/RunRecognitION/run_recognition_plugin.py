# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
'''
The run recognition plugin module will submitt he user's data to the runrecognition rest api
and then generate the html report to present to the user.
This plugin module is intended to be run from launch.sh.
'''
from restful_lib import Connection
import ConfigParser
import base64
import errno
import json
import os
import sys


def gather_and_send_data(rr_config, ts_conn, lb_conn):
    '''This is the main method that will run the plugin'''
    ts_client = TorrentServerApiClient(ts_conn)
    ts_data = _gather_ts_data(rr_config.results_id, ts_client)

    lb_client = RunRecognitionApiClient(lb_conn)

    if rr_config.generate_file:
        generate_export_file(ts_data, rr_config.site_name, rr_config.reference_genome, rr_config.application_type,
                             sys.argv[2])
        generate_export_report(sys.argv[1], sys.argv[2], rr_config.guru_api_url)
    elif (lb_client.submit_data(ts_data, rr_config.site_name, rr_config.reference_genome, rr_config.application_type,
              rr_config.guru_api_username, rr_config.guru_api_password)):
        generate_upload_report(sys.argv[1], sys.argv[2], 'Your data has been saved to the leaderboard.',
                             ts_data['experiment']['chipType'])
    else:
        generate_upload_report(sys.argv[1], sys.argv[2], 'There was an error saving your data, check to' +
                             ' make sure you provided the correct username and password.',
                             ts_data['experiment']['chipType'])


def _gather_ts_data(result_id, ts_client):
    '''connect to the ts client and gather the needed data'''
    results_data = ts_client.get_results_data(result_id)
    ts_data = {'results': results_data,
        'experiment': ts_client.get_experiment_data(results_data),
        'qualitymetrics': ts_client.get_quality_metrics(results_data),
        'libmetrics': ts_client.get_lib_metrics(results_data)}
    return ts_data


def generate_export_file(ts_data, site_name, reference_genome, application_type, out_file_path):
    '''generate the data extract file'''
    data = {}
    populate_dict_with_core_fields(data, ts_data, site_name, reference_genome, application_type, None)
    data['raw_qualitymetrics'] = ts_data['qualitymetrics']
    data['raw_libmetrics'] = ts_data['libmetrics']

    try:
        os.makedirs(out_file_path + '/files')
    except OSError, error:
        if error.errno != errno.EEXIST:
            raise

    with open(out_file_path + '/files/experiment_run.ionlb', 'w') as out_file:
        out_file.write(base64.encodestring(json.dumps(data)))

    with open(out_file_path + '/files/experiment_run_ionlb.php', 'w') as php_out_file:
        php_out_file.write('''<?php
            header('Content-disposition: attachment; filename=experiment_run.ionlb');
            header('Content-type: application/octet-stream');
            readfile('experiment_run.ionlb');
            ?>''')


def generate_export_report(template_file_path, out_file_path, guru_api_url):
    '''generates the report for when a file is exported'''
    with open(template_file_path + '/downloadInstructionsTemplate.php', 'r') as template_file:
        html_template = template_file.read()
        output = html_template.replace('@guru_base_url', guru_api_url)
        with open(out_file_path + '/RunRecognitION_block.php', 'w') as out_file:
            out_file.write(output)


def generate_upload_report(template_file_path, out_file_path, message, chip_type):
    '''
    Connects to the rest api via the given connection and creates the html page
    for the run with the given id and then serializes the html to the given file.
    '''
    with open(template_file_path + '/reportTemplate.php', 'r') as template_file:
        output = _get_output_html(template_file, message, chip_type)
        with open(out_file_path + '/RunRecognitION_block.php', 'w') as out_file:
            out_file.write(output)

    with open(template_file_path + '/leaderboard.html', 'r') as template_lb_file:
        output = _get_output_html(template_lb_file, message, chip_type)
        with open(out_file_path + '/leaderboard.html', 'w') as out_lb_file:
            out_lb_file.write(output)


def _get_output_html(template_file, message, chip_type):
    html_template = template_file.read()
    output = html_template.replace('@success_or_failure_message', message)
    output = output.replace('@chip_type', chip_type)
    return output


def _get_plugin_config(plugin_dir):
    config = ConfigParser.ConfigParser()
    config.read(plugin_dir + '/run_recognition.config')
    return config


def _get_plugin_instance_data(results_dir):
    with open(results_dir + '/startplugin.json') as json_data:
        return json.load(json_data)


class RunRecognitionPluginConfig:  # pylint: disable=R0902
    '''class to parse the configuration of this run of the plugin'''
    def __init__(self, global_config, instance_data):
        self.guru_api_url = global_config.get('rr', 'guru_api_url')
        self.torrent_server_api_url = global_config.get('rr', 'torrent_server_api_url')
        self.torrent_server_api_username = global_config.get('rr', 'torrent_server_api_username')
        self.torrent_server_api_password = global_config.get('rr', 'torrent_server_api_password')
        self.generate_file = False
        if 'tc_generatefile' in instance_data['pluginconfig']:
            self.generate_file = instance_data['pluginconfig']['tc_generatefile']
        if not self.generate_file:
            self.guru_api_username = instance_data['pluginconfig']['username']
            self.guru_api_password = instance_data['pluginconfig']['password']
        self.site_name = instance_data['pluginconfig']['site_name']
        self.application_type = instance_data['pluginconfig']['application_type']
        self.reference_genome = instance_data['pluginconfig']['reference_genome']
        self.results_id = instance_data['runinfo']['pk']


def populate_dict_with_core_fields(the_dict, ts_data, site_name, reference_genome, application_type, username):
    '''set all the core fields in to the data map'''
    the_dict['analysis_version'] = ts_data['results']['analysisVersion']
    the_dict['chip_type'] = ts_data['experiment']['chipType']
    the_dict['time_of_run'] = ts_data['results']['timeStamp']
    the_dict['jive_username'] = username
    the_dict['site_name'] = site_name
    the_dict['reference_genome'] = reference_genome
    the_dict['application_type'] = application_type


class RunRecognitionApiClient:  # pylint: disable=R0914
    '''Class to encapsulate interactions with the runrecognition api'''
    def __init__(self, connection):
        ''' init the api'''
        self.connection = connection

    def submit_data(self, ts_data, site_name, reference_genome, application_type, username, password):
        '''takes the data packages it for the leaderboard, and sends it.
        returns true on success, false on failure.'''

        # check for if there is an existing record the initial values assume it is a create
        record = self._get_existing_record(ts_data['experiment']['chipType'],
                                           ts_data['results']['timeStamp'], username)
        submission_data = {}
        submission_data['experiment_fields'] = []
        func = self.connection.request_post
        url = '/runrecognition/api/v1/experimentrun/'
        field_defs_to_add = self._get_field_definitions()
        if record is not None:
            # use the existing record as the submission data, and update the url and request type
            submission_data = record
            func = self.connection.request_put
            url += str(record['id']) + '/'

            # update the data in all of the metrics
            for cur_field in submission_data['experiment_fields']:
                cur_field_def = cur_field['field_definition']
                self._set_field_value(ts_data, cur_field, cur_field_def)

                # after the data is updated, make sure the list of metrics to
                # add no longer contains this metric
                for fd_to_add in field_defs_to_add:
                    if fd_to_add['id'] == cur_field_def['id']:
                        field_defs_to_add.remove(fd_to_add)
                        break

        # the standard fields are handled the same on create and update
        populate_dict_with_core_fields(submission_data, ts_data, site_name, reference_genome,
                                       application_type, username)

        # add in any non standard fields that need to be added
        for field_def in field_defs_to_add:
            field = {}
            field['field_definition'] = {}
            field['field_definition']['id'] = field_def['id']
            self._set_field_value(ts_data, field, field_def)
            submission_data['experiment_fields'].append(field)

        credentials = base64.encodestring('%s:%s' % (username, password)).strip()
        response = func(url, body=json.dumps(submission_data),
                        headers={'content-type': 'application/json',
                                'AUTHORIZATION': 'Basic %s' % credentials})
        status_code = response['headers'].status
        if (status_code < 200 or status_code >= 300):
            sys.stderr.write(str(response))
            return False

        return True

    @classmethod
    def _set_field_value(cls, ts_data, field, field_def):
        val = ts_data[field_def['ts_object']][field_def['ts_field']]
        if field_def['field_type'] == 'I':
            field['int_value'] = val
        elif field_def['field_type'] == 'F':
            field['float_value'] = val

    def _get_existing_record(self, chip_type, time_of_run, username):
        response = self.connection.request_get('/runrecognition/api/v1/experimentrun/',
               args={'jive_username': username,
                     'chip_type': chip_type,
                     'time_of_run': time_of_run.replace('T', ' ')},
               headers={'Accept': 'text/json'})
        rawdata = json.loads('[%s]' % response[u'body'])[0]
        count = rawdata['meta']['total_count']
        if (count > 0):
            return rawdata['objects'][0]
        else:
            return None

    def _get_field_definitions(self):
        response = self.connection.request_get('runrecognition/api/v1/experimentrunfielddefinition/',
               args={'limit': '1000'},
               headers={'Accept': 'text/json'})
        rawdata = json.loads('[%s]' % response[u'body'])[0]
        return rawdata['objects']


class TorrentServerApiClient:
    '''Class to encapsulate interactions with the torrent server api'''
    def __init__(self, connection):
        ''' init the api'''
        self.connection = connection

    def get_results_data(self, results_id):
        '''Calls the rest api to get the results data for the run'''
        response = self.connection.request_get('/rundb/api/v1/results/%s/?format=json' % results_id)
        return json.loads('[%s]' % response[u'body'])[0]

    def get_experiment_data(self, results_data):
        '''get the experiment data for the given results'''
        return self._get_associated_data(results_data[u'experiment'])

    def get_quality_metrics(self, results_data):
        '''get the quality metrics for the given results'''
        return self._get_associated_data(results_data[u'qualitymetrics'][0])

    def get_lib_metrics(self, results_data):
        '''get the quality metrics for the given results'''
        return self._get_associated_data(results_data[u'libmetrics'][0])

    def _get_associated_data(self, resource):
        '''Calls the rest api to get the associated data for the run'''
        response = self.connection.request_get('%s?format=json' % resource)
        return json.loads('[%s]' % response[u'body'])[0]


if __name__ == "__main__":
    PLUGIN_CONFIG = _get_plugin_config(sys.argv[1])
    PLUGIN_INSTANCE_DATA = _get_plugin_instance_data(sys.argv[2])
    RR_CONFIG = RunRecognitionPluginConfig(PLUGIN_CONFIG, PLUGIN_INSTANCE_DATA)
    TS_CONN = Connection(RR_CONFIG.torrent_server_api_url,
                                   RR_CONFIG.torrent_server_api_username,
                                   RR_CONFIG.torrent_server_api_password)
    LB_CONN = Connection(RR_CONFIG.guru_api_url)
    LB_CONN.h.disable_ssl_certificate_validation = True
    gather_and_send_data(RR_CONFIG, TS_CONN, LB_CONN)
