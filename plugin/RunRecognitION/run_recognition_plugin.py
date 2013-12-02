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
import requests
import urlparse
import tempfile
import shutil
import traceback


class TsConnectionError(Exception):
    ''' the exception thrown we experience issues connecting to ts to get run data '''
    pass


def gather_and_send_data(rr_config, ts_conn, lb_conn):
    '''This is the main method that will run the plugin'''
    ts_data = None
    try:
        ts_client = TorrentServerApiClient(ts_conn)
        ts_data = _gather_ts_data(rr_config.results_id, ts_client)
        lb_client = RunRecognitionApiClient(lb_conn, rr_config)
        print 'gather_and_send_data initialized'
        if rr_config.generate_file:
            print 'generating file'
            generate_export_file(ts_data, rr_config.site_name, rr_config.reference_genome, rr_config.application_type,
                                 sys.argv[2])
            generate_export_report(sys.argv[1], sys.argv[2], rr_config.guru_api_url)
        elif (lb_client.submit_data(ts_data)):
            print 'sent run to tg, about to generate upload report'
            generate_upload_report(sys.argv[1], sys.argv[2], 'Your data has been saved to the leaderboard.',
                                 ts_data['experiment']['chipType'])
        else:
            print 'failed to upload run to tg'
            message = '''
                There was an error saving your data, check to
                make sure you provided the correct username and password.
            '''
            generate_error_report(message, ts_data)
    except TsConnectionError:
        print 'caught a TsConnectionError'
        message = '''
            There was a problem connecting to Torrent Server, so we could not retrieve run data from it.
            Please verify your Run RecognitION plugin <a href="/configure/plugins/" target="_blank">configuration</a>
            and try uploading the run again.
        '''
        generate_error_report(message, ts_data)


def generate_error_report(message, ts_data=None):
    ''' create an error report for the user '''
    chip_type = "Unknown"
    if ts_data and 'experiment' in ts_data and 'chipType' in ts_data['experiment']:
        chip_type = ts_data['experiment']['chipType']
    generate_upload_report(sys.argv[1], sys.argv[2], message, chip_type)


def _gather_ts_data(result_id, ts_client):
    '''connect to the ts client and gather the needed data'''
    print 'in _gather_ts_data'
    results_data = ts_client.get_results_data(result_id)
    ts_data = {'results': results_data,
        'experiment': ts_client.get_experiment_data(results_data),
        'qualitymetrics': ts_client.get_quality_metrics(results_data),
        'libmetrics': ts_client.get_lib_metrics(results_data),
        'tfmetrics': ts_client.get_tf_metrics(results_data),
        'analysismetrics': ts_client.get_analysis_metrics(results_data)}
    if 'log' in ts_data['experiment']:
        ts_data['experiment_log'] = ts_data['experiment']['log']
    else:
        ts_data['experiment_log'] = {}
    if 'eas_set' in ts_data['experiment'] and len(ts_data['experiment']['eas_set']) > 0:
        ts_data['experiment_eas_set'] = ts_data['experiment']['eas_set'][0]
    else:
        ts_data['experiment_eas_set'] = {}
    return ts_data


def generate_export_file(ts_data, site_name, reference_genome, application_type, out_file_path):
    '''generate the data extract file'''
    print 'in generate_export_file'
    data = {}
    populate_dict_with_core_fields(data, ts_data, site_name, reference_genome, application_type, None)

    _purge_empties(ts_data['qualitymetrics'])
    _purge_empties(ts_data['libmetrics'])
    _purge_empties(ts_data['tfmetrics'])
    _purge_empties(ts_data['analysismetrics'])
    _purge_empties(ts_data['experiment_log'])
    _purge_empties(ts_data['experiment_eas_set'])
    data['raw_qualitymetrics'] = ts_data['qualitymetrics']
    data['raw_libmetrics'] = ts_data['libmetrics']
    data['raw_tfmetrics'] = ts_data['tfmetrics']
    data['raw_analysismetrics'] = ts_data['analysismetrics']
    data['raw_experiment_log'] = ts_data['experiment_log']
    data['raw_experiment_eas_set'] = ts_data['experiment_eas_set']

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


def _purge_empties(target_dict):
    print 'in _purge_empties, removing keys with empty values from dict.'
    to_remove = []
    for key, value in target_dict.items():
        if value is None or len(str(value)) == 0:
            to_remove.append(key)

    for key in to_remove:
        del target_dict[key]


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
        self.torrent_server_api_url = None
        self.torrent_server_api_username = None
        self.torrent_server_api_password = None
        self._set_ts_conn_info(global_config, instance_data)
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

    def _set_ts_conn_info(self, global_config, instance_data):
        ''' use the config the user defined or the global settings for ts connection info '''
        if 'pluginconfig' in instance_data and 'ts_url' in instance_data['pluginconfig']:
            self.torrent_server_api_url = instance_data['pluginconfig']['ts_url']
            self.torrent_server_api_username = instance_data['pluginconfig']['ts_username']
            self.torrent_server_api_password = instance_data['pluginconfig']['ts_password']
        else:
            #the ts api url has /rundb/api appended to it, which is not the
            #base path as far as tastypie is concerned, so we strip it off
            self.torrent_server_api_url = str(instance_data['runinfo']['api_url']).replace("/rundb/api", "")
            self.torrent_server_api_username = global_config.get('rr', 'torrent_server_api_username')
            self.torrent_server_api_password = global_config.get('rr', 'torrent_server_api_password')


def populate_dict_with_core_fields(the_dict, ts_data, site_name, reference_genome, application_type, username):
    '''set all the core fields in to the data map'''
    the_dict['analysis_version'] = ts_data['results']['analysisVersion']
    the_dict['chip_type'] = ts_data['experiment']['chipType']
    truncated_time = strip_tz(ts_data['results']['timeStamp'])
    the_dict['time_of_run'] = truncated_time
    the_dict['jive_username'] = username
    the_dict['site_name'] = site_name
    the_dict['reference_genome'] = reference_genome
    the_dict['application_type'] = application_type


def strip_tz(date_str):
    '''
        strip the timezone information from the date string passed in. It would be nice
        to use django.utils.timezone.localtime to do this, but that won't be available till
        django 1.4.1 comes out and we upgrade to it.
    '''
    retval = date_str
    index_of_plus = retval.find("+")
    if index_of_plus >= 0:
        retval = retval[0: index_of_plus]
    return retval


class RunRecognitionApiClient:  # pylint: disable=R0914,R0201,W0702
    '''Class to encapsulate interactions with the runrecognition api'''
    def __init__(self, connection, rr_config):
        ''' init the api'''
        self.connection = connection
        self.rr_config = rr_config

    def submit_data(self, ts_data):
        '''
            takes the data packages it for the leaderboard, and sends it.
            returns true on success, false on failure.
        '''
        # check for if there is an existing record the initial values assume it is a create
        experimentrun_id = None
        record = self._get_existing_record(ts_data['experiment']['chipType'],
                                           ts_data['results']['timeStamp'],
                                           self.rr_config.guru_api_username)
        submission_data = {}
        submission_data['experiment_fields'] = []
        func = self.connection.request_post
        url = '/runrecognition/api/v1/experimentrun/'
        field_defs_to_add = self._get_field_definitions()
        if record is not None:
            # use the existing record as the submission data, and update the url and request type
            experimentrun_id = record['id']
            submission_data = record
            func = self.connection.request_put
            url += str(record['id']) + '/'

            # update the data in all of the metrics
            for cur_field in submission_data['experiment_fields']:
                cur_field_def = cur_field['field_definition']
                if self._check_field_existence(ts_data, cur_field_def):
                    self._set_field_value(ts_data, cur_field, cur_field_def)

                # after the data is updated, make sure the list of metrics to
                # add no longer contains this metric
                for fd_to_add in field_defs_to_add:
                    if fd_to_add['id'] == cur_field_def['id']:
                        field_defs_to_add.remove(fd_to_add)
                        break

        # the standard fields are handled the same on create and update
        populate_dict_with_core_fields(submission_data, ts_data,
                                       self.rr_config.site_name,
                                       self.rr_config.reference_genome,
                                       self.rr_config.application_type,
                                       self.rr_config.guru_api_username)

        # Add in any non standard fields that need to be added. We're having to deal
        # with field_def versions now because older plugin code could not recover from
        # a field definition being absent from ts_data, so now we use a try-catch
        # to prevent future headaches.
        # pylint: disable=W0702
        for field_def in field_defs_to_add:
            field = {}
            field['field_definition'] = {}
            field['field_definition']['id'] = field_def['id']
            if self._check_field_existence(ts_data, field_def):
                self._set_field_value(ts_data, field, field_def)
                submission_data['experiment_fields'].append(field)

        temp_str = '%s:%s' % (self.rr_config.guru_api_username,
                              self.rr_config.guru_api_password)
        temp_str = temp_str.strip().encode("utf-8")
        credentials = base64.encodestring(temp_str).strip()
        response = func(url, body=json.dumps(submission_data),
                        headers={'content-type': 'application/json',
                                'AUTHORIZATION': 'Basic %s' % credentials})
        status_code = response['headers'].status
        if (status_code < 200 or status_code >= 300):
            sys.stderr.write(str(response))
            return False

        try:
            if not experimentrun_id:
                experimentrun_id = self.get_experimentrun_id(response)
            self.submit_images(experimentrun_id)
        except:
            print "Failed to submit images; Trace: %s" % traceback.format_exc()
        return True

    def submit_images(self, experimentrun_id):
        ''' submit the experimentrun's images to tg. fail silently. '''
        image_type_to_path_dict = self.get_image_dict()

        report_dir = tempfile.mkdtemp()
        report_path = os.path.join(report_dir, 'latex_report.pdf')
        if self.get_latex_pdf(self.rr_config.results_id, report_path):
            image_type_to_path_dict['pdf_report'] = report_path

        for image_type, image_path in image_type_to_path_dict.items():
            try:
                files = {'file': open(image_path, 'rb')}
                data = {'experimentrun_id': str(experimentrun_id),
                        'image_type_name': image_type}
                url = urlparse.urljoin(self.rr_config.guru_api_url, "/runrecognition/upload_experimentrun_image/")
                response = requests.post(url, files=files, data=data,
                                         auth=(self.rr_config.guru_api_username,
                                               self.rr_config.guru_api_password))
                if response.status_code != 200:
                    print "Failed to submit image\n%s\n%s" % (str(response.status_code), response.text)
            except:
                print ("Failed to submit image; Type: %s; Path: %s;\n Trace: %s"
                       % (image_type, image_path, traceback.format_exc()))
        shutil.rmtree(report_dir)

    def get_image_dict(self):
        ''' get the dictionary of image type to image path that the plugin should upload '''
        start_plugin_json = self._get_start_plugin_json_obj()
        if start_plugin_json:
            return self._get_image_dict_from_json(start_plugin_json)
        return {}

    def get_latex_pdf(self, results_id, report_file_path):
        ''' grab latex pdf from net location and save it into report_file_path '''
        try:
            url = urlparse.urljoin(self.rr_config.torrent_server_api_url, '/report/latex/%s.pdf' % str(results_id))
            report_response = requests.get(url,
                                           auth=(self.rr_config.torrent_server_api_username,
                                                 self.rr_config.torrent_server_api_password))
            if report_response.status_code >= 200 and report_response.status_code < 300:
                file_handle = open(report_file_path, 'wb+')
                for chunk in report_response.iter_content():
                    file_handle.write(chunk)
                file_handle.close()
                return True
            print '''Got back a non-success reponse while grabbing latex report.
                     Response code is: %s\nResponse Body is: %s
                  ''' % (report_response.status_code, report_response.text)
            return False
        except:
            print "Failed to grab or save latex report, error is: %s" % traceback.format_exc()
            return False

    def get_experimentrun_id(self, response):
        ''' get the saved experimentrun id from a put or post response '''
        try:
            saved_experiment = requests.get(response['headers']['location']).json()
            return saved_experiment['id']
        except:
            print "Failed to get an experiment from response: %s" % str(response)
            return None

    def _get_start_plugin_json_obj(self):
        start_plugin_json_path = os.path.join(os.getcwd(), 'startplugin.json')
        if os.path.exists(start_plugin_json_path):
            start_plugin_json_file = open(start_plugin_json_path, 'r')
            return json.loads(start_plugin_json_file.read())
        return None

    def _get_image_dict_from_json(self, start_plugin_json):
        retval = {}
        sig_proc_dir = start_plugin_json["runinfo"]["sigproc_dir"]
        basecaller_dir = start_plugin_json["runinfo"]["basecaller_dir"]
        analysis_dir = start_plugin_json["runinfo"]["analysis_dir"]
        self._update_dict_with_image(sig_proc_dir, 'Bead_density_1000.png',
                                     'bead_density_1000', retval)
        self._update_dict_with_image(basecaller_dir, 'wells_beadogram.png',
                                     'wells_beadogram', retval)
        self._update_dict_with_image(basecaller_dir, 'readLenHisto2.png',
                                     'read_len_histo2', retval)
        self._update_dict_with_image(analysis_dir, 'alignment_rate_plot.png',
                                     'alignment_rate_plot', retval)
        self._update_dict_with_image(analysis_dir, 'base_error_plot.png',
                                     'base_error_plot', retval)
        return retval

    def _update_dict_with_image(self, base_path, image_name, image_type, target_dict):
        image_path = os.path.join(base_path, image_name)
        if os.path.exists(image_path):
            target_dict[image_type] = image_path

    @classmethod
    def _check_field_existence(cls, ts_data, field_def):
        return field_def['ts_object'] in ts_data and field_def['ts_field'] in ts_data[field_def['ts_object']]

    @classmethod
    def _set_field_value(cls, ts_data, field, field_def):
        val = ts_data[field_def['ts_object']][field_def['ts_field']]
        if field_def['field_type'] == 'I':
            field['int_value'] = val
        elif field_def['field_type'] == 'F':
            field['float_value'] = val
        elif field_def['field_type'] == 'S':
            field['string_value'] = val

    def _get_existing_record(self, chip_type, time_of_run, username):
        truncated_time = strip_tz(time_of_run.replace('T', ' '))
        response = self.connection.request_get('/runrecognition/api/v1/experimentrun/',
               args={'jive_username': username.encode("utf-8"),
                     'chip_type': chip_type,
                     'time_of_run': truncated_time},
               headers={'Accept': 'text/json'})
        rawdata = json.loads('[%s]' % response[u'body'])[0]
        count = rawdata['meta']['total_count']
        if (count > 0):
            return rawdata['objects'][0]
        else:
            return None

    def _get_field_definitions(self):
        response = self.connection.request_get('runrecognition/api/v1/experimentrunfielddefinition/',
               args={'limit': '1000', 'plugin_version__lt': 1.61},
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
        return self._ts_json_get('/rundb/api/v1/results/%s/?format=json' % results_id)

    def get_experiment_data(self, results_data):
        '''get the experiment data for the given results'''
        return self._get_associated_data(results_data[u'experiment'])

    def get_quality_metrics(self, results_data):
        '''get the quality metrics for the given results'''
        return self._get_metrics(results_data, u'qualitymetrics')

    def get_lib_metrics(self, results_data):
        '''get the quality metrics for the given results'''
        return self._get_metrics(results_data, u'libmetrics')

    def get_tf_metrics(self, result_data):
        ''' get tf metrics for the given result '''
        return self._get_metrics(result_data, u'tfmetrics')

    def get_analysis_metrics(self, result_data):
        ''' get analysis for the given result '''
        return self._get_metrics(result_data, u'analysismetrics')

    def _get_metrics(self, result_data, metrics_name):
        if metrics_name in result_data and len(result_data[metrics_name]) > 0:
            return self._get_associated_data(result_data[metrics_name][0])
        return {}

    def _get_associated_data(self, resource):
        '''Calls the rest api to get the associated data for the run'''
        return self._ts_json_get('%s?format=json' % resource)

    def _ts_json_get(self, resource):
        ''' do a rest get, throw a TsConnectionError if a 200 is not returned or anything else goes wrong '''
        try:
            print 'in _ts_json_get'
            response = self.connection.request_get(resource)
            if str(response['headers']['status']) != str(200):
                raise TsConnectionError()
            json_obj = json.loads('[%s]' % response[u'body'])[0]
            print "resource is: %s, got back json:\n%s" % \
                    (resource, str(json.dumps(json_obj, sort_keys=True, indent=4, separators=(',', ': '))))
            return json_obj
        except:
            raise TsConnectionError()


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
