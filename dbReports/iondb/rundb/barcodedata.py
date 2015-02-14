# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import

import json
import os
import ast
import logging
import csv

from django.core.exceptions import ValidationError


#from django.core.serializers.json import DjangoJSONEncoder
from django.utils.functional import cached_property
from iondb.rundb.json_field import JSONEncoder

from iondb.rundb.models import Results, dnaBarcode
from iondb.rundb.plan import plan_validator

logger = logging.getLogger(__name__)

class BarcodeSampleInfo(object):
    """
    Aggregate data from plan/eas barcodesample info
    and from pipeline datasets_basecaller
    Fallback gently for non-barcoded data.

    """

    def __init__(self, resultid, result=None):
        # Gather DB info
        self.result = result or Results.objects.select_related('eas').get(pk=resultid)
        self.eas = self.result.eas

    def data(self):
        # return all data found and not filtered by pipeline
        barcodedata = self.processDatasets()

        # Helper to pull out values from datasets_basecaller structure
        # or return default values if missing
        def getPipelineValue(barcode_name, sample=None):
            if barcode_name in barcodedata:
                d = barcodedata[barcode_name]
                if sample and (d['sample'] != sample):
                    logger.warn("Overriding pipeline sample '%s' with eas value '%s'", d['sample'],sample)
                    d['sample'] = sample
            else:
                # default values if user requested sample not found in run
                d = {
                    'bam': None,
                    'barcode_name': barcode_name,
                    'sample': sample,
                    'read_count': 0,
                }
            # Pull in DB data about barcode. Return None when data not found.
            dnaBarcode = self.dnabarcodes.get(barcode_name)
            if dnaBarcode:
                for k in ['sequence', 'adapter', 'annotation', 'type']:
                    d["barcode_%s" % k] = getattr(dnaBarcode, k)

            # These default to eas values
            for k in ['reference', 'targetRegionBedFile', 'hotSpotRegionBedFile']:
                v = d.get(k)
                if (v is None) or (not v):
                    d[k] = getattr(self.eas, k)
            # Guarantee all are filled in
            for k in self._basecaller_fields:
                if k not in d:
                    d[k] = None
            return d


        data = {}
        # Process User data in plan
        #example: "barcodedSamples":"{'s1':{'barcodes': ['IonSet1_01']},'s2': {'barcodes': ['IonSet1_02']},'s3':{'barcodes': ['IonSet1_03']}}"
        for (sample, barcodes) in self.barcodedSamples.items():
            for bc in barcodes:
                data[bc] = getPipelineValue(bc, sample)

        # Add in any additional barcodes which pipeline found
        for bc in barcodedata.keys():
            if bc not in data:
                # Pipeline generated data for sample user didn't specify
                data[bc] = getPipelineValue(bc)

        # Probably redundant. Should be in barcodedata.keys()
        if 'nomatch' not in data:
            data['nomatch'] = getPipelineValue('nomatch')

        return data

    @cached_property
    def dnabarcodes(self):
        #(db_barcode.index,db_barcode.id_str,db_barcode.sequence,db_barcode.adapter,db_barcode.annotation,db_barcode.type,db_barcode.length,db_barcode.floworder)
        ret = {}
        for bc in dnaBarcode.objects.filter(name=self.eas.barcodeKitName).order_by("index"):
            ret[bc.id_str] = bc
        return ret

    @cached_property
    def barcodedSamples(self):
        barcodedSamples = self.eas.barcodedSamples
        #for both string and unicode
        if isinstance(barcodedSamples, basestring):
            #example: "barcodedSamples":"{'s1':{'barcodes': ['IonSet1_01']},'s2': {'barcodes': ['IonSet1_02']},'s3':{'barcodes': ['IonSet1_03']}}"
            try:
                barcodedSamples = json.loads(barcodedSamples)
            except ValueError as j:
                try:
                    ## What is this? Was barcodedSamples getting python string dumps?
                    barcodedSamples = ast.literal_eval(barcodedSamples)
                except Exception as e:
                    logger.fatal("Unable to parse inputs as json [%s] or python [%s]", j, e)
        try:
            for k,v in barcodedSamples.items():
                if isinstance(v['barcodes'],list):
                    for bc in v['barcodes']:
                        if not isinstance(bc,str):
                            logger.debug("INVALID bc - NOT an str - bc=%s" %(bc))
                else:
                    logger.debug("INVALID v[barcodes] - NOT a list!!! v[barcodes]=%s" %(v['barcodes']))
        except:
            logger.exception("Invalid barcodedSampleInfo value")
        return barcodedSamples

    @cached_property
    def barcodeFilter(self):
        # Read pipeline generated files (not needed, datasets_basecaller.json:read_groups/*/filtered has Include data
        barcodeFilterFile = os.path.join(self.result.get_report_path(), "basecaller_results", "barcodeFilter.txt")
        barcodeFilter = {}
        with open(barcodeFilterFile) as f:
            reader = csv.DictReader(f)
            #BarcodeId,BarcodeName,NumReads,Include
            # NB: BarcodeId is %{barcodeSetName}s_%{id}04d], BarcodeName is actually sequence
            for line in reader:
                barcodeFilter[line['BarcodeId']] = (line.Include == 1)
        return barcodeFilter


    @cached_property
    def datasetsBaseCaller(self):
        datasetsBaseCallerFile = os.path.join(self.result.get_report_path(), "basecaller_results", "datasets_basecaller.json")
        with open(datasetsBaseCallerFile) as f:
            datasetsBaseCaller = json.load(f)

        # File is tagged with a version number (yay!). Best check it.
        ver = datasetsBaseCaller.get('meta',{}).get('format_version', "0")
        if ver != "1.0":
            logger.warn("Basecaller JSON file syntax has unrecognized version: %s", ver)
        return datasetsBaseCaller

    # Stock fields in datasets_basecaller.json version 1.0
    _basecaller_fields = ['barcode_name', 'barcode_sequence', 'reference', 'sample', 'filtered', 'description']
    def processDatasets(self):
        """
        :return: json data with pipeline provided sample/barcode info
        returns basecaller_fields plus bam, read_count

        """
        ret = {}
        for item in self.datasetsBaseCaller.get('datasets', {}):
            rgs = []
            # Each file has multiple read groups. Though in practice just one.
            for idx, rg in enumerate(item.get('read_groups')):
                if idx > 1:
                    logger.warn("Multiple read_groups on single barcode not supported")
                    continue

                rgdata = self.datasetsBaseCaller.get('read_groups',{}).get(rg)
                if not rgdata:
                    logger.error("Invalid read group: %s", rg)
                    continue
                # filtered==true means rg was excluded by analysis pipeline
                if rgdata.get('filtered'):
                    continue

                # Grab these keys out of the read_groups.{{rg}}.{{k}}
                x = { 'index': idx }
                for k in self._basecaller_fields:
                    x[k] = rgdata.get(k)

                # Inconsistency in non-barcoded output
                if 'reference' not in x:
                    x['reference'] = rgdata.get('library')

                rgs.append(x)

            bcd = {
                'bam': "%s.bam" % item['file_prefix'],
                'read_count': item['read_count']
                #'read_groups': rgs,
            }
            # Primary read group only!
            if rgs:
                bcd.update(rgs[0])

            if bcd.get('barcode_name') is None:
                bcd['barcode_name'] = 'nomatch'

            # dict by name
            ret[bcd['barcode_name']] = bcd
        return ret

    def validate(self):
        err = []
        ## barcodedSamples from eas / barcodeSampleInfo from plan
        if self.barcodedSamples is None:
            return True
        for sample in self.barcodedSamples:
            err.extend(plan_validator.validate_sample_name(sample))
        if err:
            raise ValidationError(err)

        return True

    def to_json(self, **kwargs):
        return json.dumps(self.data(), allow_nan=True, cls=JSONEncoder, **kwargs)

    def to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(f, self.data(), allow_nan=True,cls=JSONEncoder)
