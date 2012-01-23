# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

"""
Mobile Views
=====

This module contains django view functions for the mobile UI
"""
import os
import fnmatch
from types import GeneratorType
from itertools import product
import csv

try:
    import json
except:
    import simplejson as json

from django import shortcuts, template
from django.utils.datastructures import SortedDict

from iondb.rundb import models

# helper method to deal with a limitation of SortedDict in Django 1.1.1
def _sorted_dict(data=None):
    ''' returns a SortedDict with given data.
    
    data can be any of legal arguments to the built-in dict() function, including generators
    '''
    if isinstance(data, GeneratorType):
        data = list(data)
    return SortedDict(data)

def _split_and_strip(str, sep, maxsplit=None):
    ''' just like str.split(sep, maxsplit), and then calling strip on each element of the resulting list '''
    return [token.strip() for token in str.split(sep, maxsplit)]

def _to_mbp(value):
    ''' display value as # megabase pairs (divide by 1000000 and format with 2 decimal points) '''
    return '%.2f' % (float(value) / 1000000)

def _as_number(value):
    ''' return a value formatted as number '''
    try:
        return '%d' % float(value)
    except ValueError:
        return value

def _as_percent(value):
    ''' return a value formatted as a percentage (multiplying by 100) '''
    pct = float(value) * 100
    if pct < 0:
        return "0%"
    elif pct > 0 and pct < 1:
        return "<1%"
    else:
        return '%.0f%%' % pct

def _value_list(dict, key_pattern, arg_tuple_iterable, mapfunc=None):
    values = [dict[key_pattern % x] for x in arg_tuple_iterable]
    return map(mapfunc, values)

class ReportHelper:
    ''' helper class that produces various report tables for use in mobile report views '''
    def __init__(self, results):
        self.results = results
        self.tabs = SortedDict((('library', 'Library'), ('tf', 'TF'), ('isp', 'ISP'), ('info', 'Info'), ('plugins', 'Plugins')))

    @property
    def versions(self):
        return self._to_sorted_dict('version.txt')

    @property
    def analysis_meta(self):
        return self._to_sorted_dict('expMeta.dat')

    @property
    def has_library_data(self):
        meta = self.analysis_meta
        return "Library" in meta and meta["Library"] != 'Library:none'

    @property
    def process_params(self):
            f = open(self.path_for_file("processParameters.txt"), 'Ur').readlines()[1:]
            return _sorted_dict((key, val) for key, val in (_split_and_strip(line, '=', 1) for line in f)
                                  if key.strip() in ('flowOrder', 'libraryKey'))

    @property
    def beadfind_wells_summary(self):
        bf_info = self.beadfind_info
        if not bf_info:
            return None

        addressable_wells = int(bf_info["Total Wells"]) - int(bf_info["Excluded Wells"])
        wells_with_isp = int(bf_info["Bead Wells"])
        live_wells = int(bf_info["Live Beads"])
        library_isp = int(bf_info["Library Beads"])
        tf_isp = int(bf_info["Test Fragment Beads"])

        wells_summary = SortedDict()
        wells_summary["Addressable"] = (_as_number(addressable_wells), u'\u00A0')
        wells_summary["&#8227; ISP Wells"] = (_as_number(wells_with_isp),
                                                    _as_percent(float(wells_with_isp) / float(addressable_wells)))
        wells_summary["&#8227; Live ISPs"] = (_as_number(live_wells),
                                                    _as_percent(float(live_wells) / float(wells_with_isp)))
        wells_summary["&#8227; TF ISPs"] = (_as_number(tf_isp),
                                                    _as_percent(float(tf_isp) / float(live_wells)))
        wells_summary["&#8227; Lib ISPs"] = (_as_number(library_isp),
                                                    _as_percent(float(library_isp) / float(live_wells)))
        return wells_summary

    @property
    def library_isp_summary(self):
        bf_info = self.beadfind_info
        if not bf_info:
            return None
        bead_summary = self._from_tsv("beadSummary.filtered.txt")

        library_isp = int(bf_info["Library Beads"])
        tf_isp = int(bf_info["Test Fragment Beads"])
        wells_with_isp = int(bf_info["Bead Wells"])
        too_short = int(bf_info["Lib Filtered Beads (read too short)"])
        keypass_fail = int(bf_info["Lib Filtered Beads (fail keypass)"])
        polyclonal = int(bf_info["Lib Filtered Beads (too many positive flows)"])
        poor_3prime_quality = int(bead_summary[0]["clipQual"])
        short_trail_adapter = int(bead_summary[0]["clipAdapter"])
        not_expected_signal = int(bf_info["Lib Filtered Beads (poor signal fit)"])
        final_reads = int(bf_info["Lib Validated Beads"])

        lib_isp_summary = SortedDict()
        lib_isp_summary["Library ISPs"] = (_as_number(library_isp),
                                                                _as_percent(float(library_isp) / float(wells_with_isp - tf_isp)))
        lib_isp_summary["&#8227; Polyclonal"] = (_as_number(polyclonal),
                                                    _as_percent(float(polyclonal) / float(library_isp)))
        lib_isp_summary["&#8227; Too short"] = (_as_number(too_short),
                                                    _as_percent(float(too_short) / float(library_isp)))
        lib_isp_summary["&#8227; Keypass failure"] = (_as_number(keypass_fail),
                                                    _as_percent(float(keypass_fail) / float(library_isp)))
        lib_isp_summary["&#8227; Poor Signal Profile"] = (_as_number(not_expected_signal),
                                                    _as_percent(float(not_expected_signal) / float(library_isp)))
        lib_isp_summary["&#8227; 3&prime; Adapter trim"] = (_as_number(short_trail_adapter),
                                                    _as_percent(float(short_trail_adapter) / float(library_isp)))
        lib_isp_summary["&#8227; 3&prime; Quality trim"] = (_as_number(poor_3prime_quality),
                                                    _as_percent(float(poor_3prime_quality) / float(library_isp)))
        lib_isp_summary["&#8227; Library Reads"] = (_as_number(final_reads),
                                                    _as_percent(float(final_reads) / float(library_isp)))
        return lib_isp_summary

    @property
    def beadfind_info(self):
        bf_file = self.get_report_layout()["Beadfind"]["file_path"]
        if self.has_file(bf_file):
            f = open(self.path_for_file(bf_file), 'Ur').readlines()[1:]
            return _sorted_dict(_split_and_strip(line, '=', 1) for line in f)

    @property
    def quality_summary(self):
        return self._to_sorted_dict(self.get_report_layout()["Quality Summary"]["file_path"])

    @property
    def alignment_summary(self):
        as_file = self.get_report_layout()["Alignment Summary"]["file_path"]
        if self.has_file(as_file):
            return self._to_sorted_dict(as_file)

    @property
    def has_full_alignment_info(self):
        return 'Total number of Sampled Reads' not in self.alignment_summary

    def genome_summary(self):
        alignment_summary = self.alignment_summary
        if not alignment_summary:
            return None

        metrics = self.get_report_layout()["Alignment Summary"]["pre_metrics"]
        genome_summary = SortedDict()
        for label, metric in metrics.items():
            if isinstance(metric, (list, tuple)):
                value = '%d %s' % (float(alignment_summary[metric[0]]), metric[1])
            else:
                value = alignment_summary[metric]
            genome_summary[label] = value
        return genome_summary

    def tf_helpers(self):
        return [TFHelper(tf) for tf in self.results.tfmetrics_set.all()]

    @property
    def alignment_independent_summary(self):
        quality_summary = self.quality_summary
        if not quality_summary:
            return None

        indep_summary = SortedDict()
        indep_summary['Total Number of Bases [Mbp]'] = _to_mbp(quality_summary['Number of Bases at Q0'])
        indep_summary['&#8227; Number of Q17 Bases [Mbp]'] = _to_mbp(quality_summary['Number of Bases at Q17'])
        indep_summary['&#8227; Number of Q20 Bases [Mbp]'] = _to_mbp(quality_summary['Number of Bases at Q20'])
        indep_summary['Total Number of Reads'] = quality_summary['Number of Reads at Q0']
        indep_summary['Mean Length [bp]'] = quality_summary['Mean Read Length at Q0']
        indep_summary['Longest Read [bp]'] = quality_summary['Max Read Length at Q0']

        return indep_summary

    @property
    def full_library_alignment_summary(self):
        alignment_summary = self.alignment_summary
        if not alignment_summary:
            return None

        def value_list(key_pattern, mapfunc=_as_number):
            return _value_list(alignment_summary, key_pattern, ('Q17', 'Q20','Q47'), mapfunc)


        full_lib_summary = SortedDict()
        full_lib_summary['Total # Bases [Mbp]'] = value_list('Filtered Mapped Bases in %s Alignments', _to_mbp)
        full_lib_summary['Mean Length [bp]'] = value_list('Filtered %s Mean Alignment Length')
        full_lib_summary['Longest Alignment [bp]'] = value_list('Filtered %s Longest Alignment')
        full_lib_summary['Mean Coverage Depth'] = value_list('Filtered %s Mean Coverage Depth')
        full_lib_summary['% of Library Covered'] = value_list('Filtered %s Coverage Percentage')

        return full_lib_summary

    @property
    def sampled_library_alignment_summary(self):
        alignment_summary = self.alignment_summary
        if not alignment_summary:
            return None

        def value_list(key_pattern, mapfunc=_as_number):
            return _value_list(alignment_summary, key_pattern,
                               product(('Sampled', 'Extrapolated'), ('Q17', 'Q20','Q47')), mapfunc)

        sampled_lib_summary = SortedDict()
        sampled_lib_summary['Total # Bases [Mbp]'] = value_list('%s Filtered Mapped Bases in %s Alignments', _to_mbp)
        sampled_lib_summary['Mean Length [bp]'] = value_list('%s Filtered %s Mean Alignment Length')
        sampled_lib_summary['Longest Alignment [bp]'] = value_list('%s Filtered %s Longest Alignment')
        sampled_lib_summary['Mean Coverage Depth'] = value_list('%s Filtered %s Mean Coverage Depth')
        sampled_lib_summary['% of Library Covered'] = value_list('%s Filtered %s Coverage Percentage')

        return sampled_lib_summary

    def read_alignment_distribution_info(self):
        distr_file = "alignTable.txt"
        if not self.has_file(distr_file):
            return None

        distribution = self._from_tsv(distr_file)
        filtered_distribution = [row for row in distribution
                                 if sum(int(v) for k, v in row.iteritems() if k and k != 'readLen') > 0]
        info = SortedDict()
        key_labels = SortedDict((('readLen', 'Read Length[bp]'), ('nread', 'Reads'), ('unalign', 'Unmapped'), ('excluded', 'Excluded'),
                      ('clipped', 'Clipped'), ('err0', 'Perfect'), ('err1', '1 mismatch'), ('err2', '&ge;2 mismatches')))
        
        for key, label in key_labels.iteritems():
            vals = [row[key] for row in filtered_distribution]
            info[label] = vals
        return info

    def plugin_links(self):
        plugin_path = self.path_for_file('plugin_out')
        links = { }
        for plugin_item in os.listdir(plugin_path):
            plugin_item_path = os.path.join(plugin_path, plugin_item)
            if os.path.isdir(plugin_item_path):
                links.update([(os.path.join('plugin_out', plugin_item, html_file), html_file)
                              for html_file in fnmatch.filter(os.listdir(plugin_item_path), '*.html')])
        return links

    def progress(self):
        progress_file = "progress.txt"
        if not self.has_file(progress_file):
            return None

        progress_raw = self._to_sorted_dict(progress_file)
        progress_tasks = {"wellfinding": "Well Characterization",
                           "signalprocessing": "Signal Processing",
                           "basecalling": "Basecalling",
                           "sffread": "Creating Fastq",
                           "alignment": "Aligning Reads"
                         }
        progress_meaning = {"yellow": "In Progress", "grey": "Not Yet Started"}
        return _sorted_dict((progress_tasks[k], progress_meaning[v]) for k, v in progress_raw.iteritems() if v != 'green')

    def get_report_layout(self):
        with self.open_report_file('report_layout.json') as f:
            return json.load(f)

    def _to_sorted_dict(self, filename):
        ''' read a file which has a bunch of lines in the format
        
        key = value
        
        and return a SortedDict with key->value mappings, sorted in the order of the file
        '''
        with self.open_report_file(filename) as f:
            return _sorted_dict(_split_and_strip(line, '=', 1) for line in f)

    def _from_tsv(self, filename):
        """read a tab-separated file, with a header line, and return an array of dicts.
        
        each dict in the array corresponds to a line in the file, mapping field names from header to values
        """
        with self.open_report_file(filename) as f:
            dr = csv.DictReader(f, delimiter='\t')
            return [row for row in dr]

    def path_for_file(self, filename):
        ''' returns path for given file in report directory '''
        return '/'.join(self.results.get_report_dir().split('/') + [filename])

    def link_for_file(self, filename):
        ''' returns absolute (relative to server root) path for linking to a file '''
        return '/'.join(self.results.reportLink.split('/')[:-1] + [filename])

    def has_file(self, filename):
        return os.path.exists(self.path_for_file(filename))

    def open_report_file(self, filename, mode='r'):
        return open(self.path_for_file(filename), mode)

class TFHelper:
    ''' helper class for TF detail in mobile report views '''
    def __init__(self, tf):
        self.tf = tf

    def percent_50Q10(self):
        return _as_percent(self.tf.Q17ReadCount / self.tf.keypass)

    def detail_link(self):
        return '#tf_%d' % self.tf.pk

    def quality_metrics(self):
        return _sorted_dict((label, getattr(self.tf, key, None)) for label, key in
                            (('TF Seq', 'sequence'), ('Num', 'keypass'), ('Avg Q17 read length', 'Q17Mean'), ('50AQ17', 'Q17ReadCount')))

def runs(request):
    """ Show last 30 runs"""
    runs = models.Experiment.objects.all().select_related().order_by('-date')[0:30]
    context = template.RequestContext(request, {"runs":runs})
    return shortcuts.render_to_response("rundb/mobile/runs.html",
                                        context_instance=context)

def run(request, run_pk):
    """ Show reports for run with given pk, when there is more than one result """
    exp = shortcuts.get_object_or_404(models.Experiment, pk=run_pk)
    context = template.RequestContext(request, {"exp":exp})
    return shortcuts.render_to_response("rundb/mobile/reports.html",
                                        context_instance=context)

def report(request, result_pk):
    """ Show report details for result with given pk """
    result = shortcuts.get_object_or_404(models.Results, pk=result_pk)
    helper = ReportHelper(result)
    context = template.RequestContext(request, {"result":result, "helper": helper})
    template_name = "rundb/mobile/%s.html" % ('report_progress' if helper.progress() else 'report')
    return shortcuts.render_to_response(template_name, context_instance=context)

