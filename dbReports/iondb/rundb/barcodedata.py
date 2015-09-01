# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import

import json
import os
import ast
import logging
import csv

from django.core.exceptions import ValidationError
from django.conf import settings
#from django.core.serializers.json import DjangoJSONEncoder
from django.utils.functional import cached_property
from iondb.rundb.json_field import JSONEncoder
from iondb.rundb.models import Results, dnaBarcode
from iondb.rundb.plan import plan_validator

logger = logging.getLogger(__name__)

REFERENCE = 'reference'
REFERENCE_FULL_PATH = 'referenceFullPath'
TARGET_REGION_BED = 'targetRegionBedFile'
HOT_SPOT_BED = 'hotSpotRegionBedFile'
BAM = 'bam'
BAM_FULL_PATH = 'bamFullPath'
NON_BARCODED = 'nonbarcoded'

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

    def data(self, start_json=None):
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
                logger.warn("Requested barcode name %s not found, using given sample %s with no bam data", barcode_name, sample)
                # default values if user requested sample not found in run
                d = {
                    BAM: None,
                    'barcode_name': barcode_name,
                    'sample': sample,
                    'read_count': 0,
                }

            # Pull in DB data about barcode. Return None when data not found.
            if barcode_name != NON_BARCODED:
                dnaBarcode = self.dnabarcodes.get(barcode_name)
                if dnaBarcode:
                    for k in ['sequence', 'adapter', 'annotation', 'type']:
                        d["barcode_%s" % k] = getattr(dnaBarcode, k)
                else:
                    logger.error("Barcode given does not exist in TS database: %s", barcode_name)

            # These default to eas values
            for k in [REFERENCE, TARGET_REGION_BED, HOT_SPOT_BED]:
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
        for (sample, info) in self.barcodedSamples.items():
            for bc in info.get('barcodes'):
                data[bc] = getPipelineValue(bc)
                # Override pipeline with plan supplied values.
                if sample != data[bc].get('sample'):
                    logger.warn("Overriding pipeline sample '%s' with eas value '%s'", data[bc]['sample'],sample)
                    data[bc]['sample'] = sample
                # nucleotideType?
                plan_barcodeSampleInfo = info.get('barcodeSampleInfo', {}).get(bc)
                if plan_barcodeSampleInfo:
                    # Barcode Specific reference data from plan
                    for k in [REFERENCE, TARGET_REGION_BED, HOT_SPOT_BED]:
                        if k in plan_barcodeSampleInfo:
                            data[bc][k] = plan_barcodeSampleInfo[k]

        # Add in any additional barcodes which pipeline found
        for bc in barcodedata.keys():
            if bc in data:
                continue
            if barcodedata[bc].get('filtered', False):
                continue
            # Pipeline generated data for sample user didn't specify
            data[bc] = getPipelineValue(bc)
            #logger.debug("Pipeline found extra barcode [%s]: %s", bc, data[bc])

        # Probably redundant. Should be in barcodedata.keys()
        if 'nomatch' not in data and NON_BARCODED not in data:
            data['nomatch'] = getPipelineValue('nomatch')
            logger.debug("Generated placeholder entry for nomatch: %s", data['nomatch'])

        # add full path values for a series of entriess
        reportFullPath = self.result.get_report_path()
        for currentBarcode in data.values():
            if BAM in currentBarcode:
                currentBarcode[BAM_FULL_PATH] = os.path.join(reportFullPath, currentBarcode[BAM]) if currentBarcode[BAM] else ""
            if start_json is not None and REFERENCE in currentBarcode and currentBarcode[REFERENCE]:
                refName = currentBarcode[REFERENCE]
                currentBarcode[REFERENCE_FULL_PATH] = os.path.join('/results', 'referenceLibrary', start_json['runinfo']['tmap_version'],refName, "%s.fasta" % refName)
            else:
                currentBarcode[REFERENCE_FULL_PATH] = ''

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
        if isinstance(barcodedSamples, basestring):

            try:
                barcodedSamples = json.loads(barcodedSamples, encoding=settings.DEFAULT_CHARSET)
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
                            logger.error("INVALID bc - NOT an str - bc=%s" %(bc))
                else:
                    logger.error("INVALID v[barcodes] - NOT a list!!! v[barcodes]=%s" %(v['barcodes']))
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
            datasetsBaseCaller = json.load(f, encoding=settings.DEFAULT_CHARSET)

        # File is tagged with a version number (yay!). Best check it.
        ver = datasetsBaseCaller.get('meta',{}).get('format_version', "0")
        if ver != "1.0":
            logger.warn("Basecaller JSON file syntax has unrecognized version: %s", ver)
        return datasetsBaseCaller

    # Stock fields in datasets_basecaller.json version 1.0
    _basecaller_fields = ['barcode_name', 'barcode_sequence', REFERENCE, 'sample', 'filtered', 'description']
    def processDatasets(self):
        """
        :return: json data with pipeline provided sample/barcode info
        returns basecaller_fields plus bam, read_count

        """

        # for the moment we are attempting to detect an un-barcoded dataset by the curious absence of the barcode_config/barcode_id node
        isUnbarcoded = False
        barcode_config = self.datasetsBaseCaller.get("barcode_config")
        if barcode_config:
            barcode_id = barcode_config.get("barcode_id")
            isUnbarcoded = barcode_id is None
        else:
            isUnbarcoded = True

        ret = {}
        # we are going to handle unbarcoded data as a special case where as a single item is generated
        if isUnbarcoded:
            singleDataset   = self.datasetsBaseCaller.get('datasets')[0]
            singleReadGroup = self.datasetsBaseCaller.get('read_groups').itervalues().next()
            unbarcoded = {}
            unbarcoded['bam'] = singleDataset['basecaller_bam']
            unbarcoded['barcode_sequence'] = ''
            unbarcoded[REFERENCE] = singleReadGroup[REFERENCE]
            unbarcoded['sample'] = singleReadGroup['sample']
            unbarcoded['filtered'] = singleReadGroup['filtered']
            unbarcoded['description'] = singleReadGroup['description']
            unbarcoded['barcode_name'] = NON_BARCODED
            unbarcoded['filtered'] = False
            unbarcoded['index'] = 0
            unbarcoded['read_count'] = singleReadGroup['read_count']
            ret[NON_BARCODED] = unbarcoded
            return ret


        for item in self.datasetsBaseCaller.get('datasets', {}):
            rgs = []
            # Each file has multiple read groups. Though in practice just one.
            for idx, rg in enumerate(item.get('read_groups',[])):
                if idx > 1:
                    logger.warn("Multiple read_groups on single barcode not supported")
                    break

                # Get matching section from top level read_groups section
                rgdata = self.datasetsBaseCaller.get('read_groups',{}).get(rg)
                if not rgdata:
                    logger.error("Invalid read group: %s", rg)
                    continue
                # Grab these keys out of the read_groups.{{rg}}.{{k}}
                x = { 'index': idx }
                for k in self._basecaller_fields:
                    x[k] = rgdata.get(k)

                # Inconsistency in non-barcoded output
                if REFERENCE not in x:
                    x[REFERENCE] = rgdata.get('library')

                rgs.append(x)
                break

            bcd = {
                BAM : "%s.bam" % item['file_prefix'],
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
