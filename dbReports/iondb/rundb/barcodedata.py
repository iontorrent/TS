# Copyright (C) 2014 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import absolute_import

import json
import os
import ast
import logging
from collections import namedtuple

import iondb.bin.dj_config
from django.conf import settings
from django.utils.functional import cached_property
from iondb.rundb.models import Results, dnaBarcode, ReferenceGenome, Experiment

logger = logging.getLogger(__name__)

REFERENCE = "reference"
REFERENCE_FULL_PATH = "reference_fullpath"
REFERENCE_FULL_NAME = "reference_full_name"
TARGET_REGION_FILEPATH = "target_region_filepath"
TARGET_REGION_BED = "targetRegionBedFile"
HOT_SPOT_FILE_PATH = "hotspot_filepath"
HOT_SPOT_BED = "hotSpotRegionBedFile"
SSE_BED_FILEPATH = "sse_filepath"
SSE_BED = "sseBedFile"
BAM = "bam_file"
BAM_FULL_PATH = "bam_filepath"
NON_BARCODED = "nonbarcoded"
NO_MATCH = "nomatch"
SAMPLE = "sample"
SAMPLE_ID = "sample_id"
FILTERED = "filtered"
BARCODE_DESCRIPTION = "barcode_description"
DESCRIPTION = "description"
BARCODE_NAME = "barcode_name"
BARCODE_INDEX = "barcode_index"
INDEX = "index"
READ_COUNT = "read_count"
ALIGNED = "aligned"
BARCODE_ADAPTER = "barcode_adapter"
BARCODE_ANNOTATION = "barcode_annotation"
BARCODE_SEQUENCE = "barcode_sequence"
BARCODE_TYPE = "barcode_type"
NUCLEOTIDE_TYPE = "nucleotide_type"
CONTROL_SEQUENCE_TYPE = "control_sequence_type"
GENOME_URL = "genome_urlpath"
SAMPLE_NUCLEOTIDE_TYPE = "nucleotideType"
SAMPLE_CONTROL_SEQEUENCE_TYPE = "controlSequenceType"
NTC_CONTROL = "control_type"

END_BARCODE_KIT_NAME = "end_barcode_kit_name"
END_BARCODE_NAME = "end_barcode_name"
END_BARCODE_INDEX = "end_barcode_index"
END_BARCODE_ADAPTER = "end_barcode_adapter"
END_BARCODE_ANNOTATION = "end_barcode_annotation"
END_BARCODE_SEQUENCE = "end_barcode_sequence"
END_BARCODE_TYPE = "end_barcode_type"
DUAL_BARCODE_NAME = "dual_barcode_name"

END_BARCODE = "endBarcode"

DATASET_BARCODE = "barcode"
DATASET_END_BARCODE = "end_barcode"

# TODO: for later release
# ANALYSIS_PARAMETERS = 'analysis_parameters'

DnaBarcodeModel = namedtuple(
    "DnaBarcodeModel",
    [
        "sequence",
        "adapter",
        "index",
        "annotation",
        "type",
        "end_sequence",
        "end_adapter",
    ],
)

WARN_BARCODE_NOT_FOUND = "dnaBarcode query failed to find any result: dnaBarcode(name={barcodeKit}, id_str={barcodeName}"
ERROR_BARCODE_NOT_FOUND = "No barcode data for {barcodeName} in kit: {barcodeKit}."


def get_reference(short_name, tmap_version):
    reference = ReferenceGenome.objects.filter(
        short_name=short_name, enabled=True, index_version=tmap_version
    ).first()
    if not reference:
        reference = ReferenceGenome.objects.filter(short_name=short_name).first()
    return reference


class BarcodeSampleInfo(object):

    """
    Aggregate data from plan/eas barcodesample info
    and from pipeline datasets_basecaller
    Fallback gently for non-barcoded data.
    """

    def __init__(self, resultid, result=None):
        # Gather DB info
        self.result = result or Results.objects.select_related("eas").get(pk=resultid)
        self.eas = self.result.eas

    @staticmethod
    def getFullBamPath(reportFullPath, referenceGenome, bamFileName, filtered):
        """
        Helper method to find the full path of the BAM file
        :param: reportFullPath:
        :param referenceGenome:
        :param bamFileName:
        :param filtered: Boolean indicating if the item was filtered
        :return:
        """
        bamFullPath = ""
        if referenceGenome and not filtered:
            bamFullPath = os.path.join(reportFullPath, bamFileName)
        else:
            # if there is not a reference file then we need to file the basecaller bam file
            if not bamFileName.endswith(".basecaller.bam"):
                bamFileName = bamFileName.replace(".bam", ".basecaller.bam")
            bamFullPath = os.path.join(
                reportFullPath, "basecaller_results", bamFileName
            )

        # sanity check to make sure we do not reference files which don't exist
        if not os.path.exists(bamFullPath):
            logger.warn("Could not find bam file in expected location: " + bamFullPath)

        return bamFullPath

    def dataNonBarcoded(self):
        """
        Generates a "barcodes" dataset for non-barcoded data
        :param start_json:
        :return:
        """
        data = dict()
        reportFullPath = self.result.get_report_path()
        # the data structure in this dictionary should be that both the datasets and read_groups for
        # unbarcoded samples should have one and only one instance contained within
        singleDataset = self.datasetsBaseCaller.get("datasets")[0]
        singleReadGroup = iter(
            self.datasetsBaseCaller.get("read_groups").values()
        ).next()
        unbarcodedEntry = dict()

        # generate the reference genome information
        referenceGenome = self.eas.reference
        tmap_version = iondb.bin.dj_config.get_tmap_version()
        samples = self.result.experiment.samples.all()
        referenceObj = get_reference(referenceGenome, tmap_version)

        unbarcodedEntry[REFERENCE] = referenceGenome
        unbarcodedEntry[REFERENCE_FULL_PATH] = (
            os.path.join(referenceObj.reference_path, "%s.fasta" % referenceGenome)
            if referenceObj
            else ""
        )
        unbarcodedEntry[REFERENCE_FULL_NAME] = referenceObj.name if referenceObj else ""
        unbarcodedEntry[ALIGNED] = (REFERENCE in unbarcodedEntry) and bool(
            unbarcodedEntry[REFERENCE]
        )
        unbarcodedEntry[BAM] = (
            singleDataset["basecaller_bam"]
            if not unbarcodedEntry[REFERENCE]
            else "rawlib.bam"
        )
        unbarcodedEntry[FILTERED] = (
            singleReadGroup[FILTERED] if FILTERED in singleReadGroup else False
        )
        unbarcodedEntry[BAM_FULL_PATH] = BarcodeSampleInfo.getFullBamPath(
            reportFullPath,
            unbarcodedEntry[REFERENCE],
            unbarcodedEntry[BAM],
            unbarcodedEntry[FILTERED],
        )
        unbarcodedEntry[HOT_SPOT_FILE_PATH] = (
            singleDataset[HOT_SPOT_BED]
            if HOT_SPOT_BED in singleDataset
            else self.eas.hotSpotRegionBedFile
        )
        unbarcodedEntry[READ_COUNT] = singleReadGroup[READ_COUNT]
        unbarcodedEntry[TARGET_REGION_FILEPATH] = self.eas.targetRegionBedFile
        unbarcodedEntry[SSE_BED_FILEPATH] = self.eas.sseBedFile
        unbarcodedEntry[NUCLEOTIDE_TYPE] = ""
        unbarcodedEntry[CONTROL_SEQUENCE_TYPE] = ""
        unbarcodedEntry[SAMPLE] = singleReadGroup[SAMPLE]
        unbarcodedEntry[SAMPLE_ID] = (
            samples.first().externalId if samples.count() == 1 else ""
        )
        unbarcodedEntry[NUCLEOTIDE_TYPE] = ""
        unbarcodedEntry[NTC_CONTROL] = ""
        data[NON_BARCODED] = unbarcodedEntry

        # construct the reference genome url which is web accessible
        unbarcodedEntry[GENOME_URL] = (
            "/auth/output/"
            + tmap_version
            + "/"
            + unbarcodedEntry[REFERENCE]
            + "/"
            + unbarcodedEntry[REFERENCE]
            + ".fasta"
            if unbarcodedEntry[REFERENCE]
            else ""
        )

        return data

    def dataBarcoded(self):
        """
        Gets data set for barcoded data
        :param start_json:
        :return:
        """

        def getBarcodeSampleFromEAS(EASbarcodedSamples, barcodeName):
            """
            This helper method will look for the barcode information within the EAS barcodeSamples data structure embedded in dicitonary
            :param EASbarcodedSamples:
            :param barcodeName:
            :return:
            """
            for sample in list(EASbarcodedSamples.keys()):
                barcodeSampleTop = EASbarcodedSamples[sample]
                if barcodeName in barcodeSampleTop["barcodes"]:
                    return (
                        sample,
                        barcodeSampleTop["barcodeSampleInfo"][barcodeName],
                        barcodeSampleTop.get("dualBarcodes", []),
                    )
            raise Exception(
                "Cannot find the barcode data from EAS barcodeSamples json structure."
            )

        def createDnaBarcodeModel(read_group={}):
            """
            creating a named tupe micmicking dnaBarcode model

            read_group["barcode"] and read_group["end_barcode"] are added in 5.10

            :param read_group:
            :return:
            """
            if not read_group:
                return None

            if DATASET_BARCODE in read_group:
                sequence = read_group.get(DATASET_BARCODE).get(BARCODE_SEQUENCE, "")
                adapter = read_group.get(DATASET_BARCODE).get(BARCODE_ADAPTER, "")
            else:
                sequence = read_group.get(BARCODE_SEQUENCE, "")
                adapter = read_group.get(BARCODE_ADAPTER, "")

            end_sequence = ""
            end_adapter = ""
            if DATASET_END_BARCODE in read_group:
                # read_group["barcode"] and read_group["end_barcode"] has the same field name
                end_sequence = read_group.get(DATASET_END_BARCODE).get(BARCODE_SEQUENCE)
                end_adapter = read_group.get(DATASET_END_BARCODE).get(BARCODE_ADAPTER)

            return DnaBarcodeModel(
                sequence=sequence,
                adapter=adapter,
                index=read_group.get(INDEX, 0),
                type="",
                annotation="",
                end_sequence=end_sequence,
                end_adapter=end_adapter,
            )
            
        def getBarcodeName(readGroupName):
            # many times it seems that the readGroupId is a join between the results runid and barcodeName
            # so they need to break this out and parse the readGroupId
            if "." in readGroupName:
                runid, barcodeName = readGroupName.split(".", 1)
            else:
                barcodeName = readGroupName
            return barcodeName

        data = dict()
        reportFullPath = self.result.get_report_path()
        # now we are going to handle barcode data and get all of the primary sources of information
        basecallerResults = self.datasetsBaseCaller
        EASbarcodedSamples = self.barcodedSamples
        tmap_version = iondb.bin.dj_config.get_tmap_version()

        include_filtered = getattr(settings, "PLUGINS_INCLUDE_FILTERED_BARCODES", False)

        for dataset in basecallerResults.get("datasets"):
            barcodeEntry = dict()
            readGroups = dataset["read_groups"]
            
            # Repeat Sequencing stores more than one read group in one dataset - iterate over read groups
            if len(readGroups) == 0:
                logger.warn("No read_groups found in dataset " + dataset.get("dataset_name", ""))
                continue
            
            barcodeName = getBarcodeName(readGroups[0])
            if NO_MATCH in barcodeName:
                continue
            
            # Infer filtered status & verify all read groups belong to the same barcode
            is_filtered  = True
            same_barcode = True
            for readGroupName in readGroups:
                is_filtered = is_filtered and basecallerResults.get("read_groups")[readGroupName].get(FILTERED, False)
                same_barcode = same_barcode and barcodeName == getBarcodeName(readGroupName)
            
            if not same_barcode:
                logger.warn("Different barcodes in the same dataset are not supported")
                continue
            
            if is_filtered and not include_filtered:
                continue
            
            barcodeEntry[FILTERED] = is_filtered
            barcodeEntry[READ_COUNT] = dataset.get("read_count", 0)
            firstReadGroup = basecallerResults.get("read_groups")[readGroups[0]]

            try:
                dnaBarcodeData = dnaBarcode.objects.get(
                    name=self.eas.barcodeKitName, id_str=barcodeName
                )
            except dnaBarcode.DoesNotExist:
                logger.warn(
                    WARN_BARCODE_NOT_FOUND.format(
                        barcodeName=barcodeName, barcodeKit=self.eas.barcodeKitName
                    )
                )
                dnaBarcodeData = createDnaBarcodeModel(read_group=firstReadGroup)

            if not dnaBarcodeData:
                logger.error(
                    ERROR_BARCODE_NOT_FOUND.format(
                        barcodeName=barcodeName, barcodeKit=self.eas.barcodeKitName
                    )
                )
                continue

            barcodeEntry[REFERENCE] = self.eas.reference
            barcodeEntry[BARCODE_ANNOTATION] = dnaBarcodeData.annotation
            barcodeEntry[BARCODE_TYPE] = dnaBarcodeData.type

            barcodeEntry[BARCODE_SEQUENCE] = dnaBarcodeData.sequence
            barcodeEntry[BARCODE_ADAPTER] = dnaBarcodeData.adapter
            barcodeEntry[BARCODE_NAME] = barcodeName
            barcodeEntry[BARCODE_INDEX] = dnaBarcodeData.index

            # these can be overridden by the sample barcode data
            barcodeEntry[TARGET_REGION_FILEPATH] = self.eas.targetRegionBedFile
            barcodeEntry[HOT_SPOT_FILE_PATH] = self.eas.hotSpotRegionBedFile
            barcodeEntry[SSE_BED_FILEPATH] = self.eas.sseBedFile

            # in order to get information out of the EAS.barcodedSamples data structure there needs to be
            # a reverse lookup since the barcode name is a child of the sample id
            barcodeEntry[SAMPLE] = ""
            barcodeEntry[NUCLEOTIDE_TYPE] = ""
            barcodeEntry[CONTROL_SEQUENCE_TYPE] = ""
            barcodeEntry[BARCODE_DESCRIPTION] = ""
            barcodeEntry[NTC_CONTROL] = ""

            # dual barcoding
            barcodeEntry[END_BARCODE_NAME] = ""
            barcodeEntry[END_BARCODE_INDEX] = ""
            barcodeEntry[END_BARCODE_ADAPTER] = ""
            barcodeEntry[END_BARCODE_ANNOTATION] = ""
            barcodeEntry[END_BARCODE_SEQUENCE] = ""
            barcodeEntry[END_BARCODE_TYPE] = ""
            barcodeEntry[DUAL_BARCODE_NAME] = ""
            barcodeEntry[END_BARCODE_KIT_NAME] = self.eas.endBarcodeKitName

            if dnaBarcodeData.end_sequence:
                barcodeEntry[END_BARCODE_NAME] = barcodeName
                barcodeEntry[DUAL_BARCODE_NAME] = barcodeName
                barcodeEntry[END_BARCODE_INDEX] = dnaBarcodeData.index
                barcodeEntry[END_BARCODE_ANNOTATION] = dnaBarcodeData.annotation
                barcodeEntry[END_BARCODE_TYPE] = dnaBarcodeData.type

                barcodeEntry[END_BARCODE_ADAPTER] = dnaBarcodeData.end_adapter
                barcodeEntry[END_BARCODE_SEQUENCE] = dnaBarcodeData.end_sequence

            if len(EASbarcodedSamples) > 0:
                # attempt to find the barcode in the EAS BarcodeSample mapping
                try:
                    sample, barcodedSample, dualBarcodes = getBarcodeSampleFromEAS(
                        EASbarcodedSamples, barcodeName
                    )
                    barcodeEntry[SAMPLE] = sample
                    barcodeEntry[NUCLEOTIDE_TYPE] = barcodedSample.get(
                        SAMPLE_NUCLEOTIDE_TYPE, barcodeEntry[NUCLEOTIDE_TYPE]
                    )
                    barcodeEntry[CONTROL_SEQUENCE_TYPE] = barcodedSample.get(
                        SAMPLE_CONTROL_SEQEUENCE_TYPE,
                        barcodeEntry[CONTROL_SEQUENCE_TYPE],
                    )
                    barcodeEntry[BARCODE_DESCRIPTION] = barcodedSample.get(
                        DESCRIPTION, barcodeEntry[BARCODE_DESCRIPTION]
                    )
                    barcodeEntry[TARGET_REGION_FILEPATH] = barcodedSample.get(
                        TARGET_REGION_BED, barcodeEntry[TARGET_REGION_FILEPATH]
                    )
                    barcodeEntry[HOT_SPOT_FILE_PATH] = barcodedSample.get(
                        HOT_SPOT_BED, barcodeEntry[HOT_SPOT_FILE_PATH]
                    )
                    barcodeEntry[REFERENCE] = barcodedSample.get(
                        REFERENCE, barcodeEntry[REFERENCE]
                    )
                    barcodeEntry[NTC_CONTROL] = barcodedSample.get(
                        "controlType", barcodeEntry[NTC_CONTROL]
                    )
                    if SSE_BED in barcodedSample:
                        barcodeEntry[SSE_BED_FILEPATH] = barcodedSample[SSE_BED]
                    elif (
                        barcodeEntry[TARGET_REGION_FILEPATH]
                        != self.eas.targetRegionBedFile
                    ):
                        barcodeEntry[SSE_BED_FILEPATH] = ""

                    # dynamic dual barcoding
                    if self.eas.endBarcodeKitName:
                        sampleEndBarcode = barcodedSample.get(
                            END_BARCODE, barcodeEntry[END_BARCODE_NAME]
                        )
                        barcodeEntry[END_BARCODE_NAME] = sampleEndBarcode
                        if sampleEndBarcode:
                            try:
                                endBarcodeObj = dnaBarcode.objects.get(
                                    name=self.eas.endBarcodeKitName,
                                    id_str=sampleEndBarcode,
                                )
                            except dnaBarcode.DoesNotExist:
                                logger.warn(
                                    WARN_BARCODE_NOT_FOUND.format(
                                        barcodeName=barcodeName,
                                        barcodeKit=self.eas.barcodeKitName,
                                    )
                                )
                                endBarcodeObj = createDnaBarcodeModel(
                                    read_group=singleReadGroup
                                )

                            if endBarcodeObj:
                                barcodeEntry[
                                    END_BARCODE_ADAPTER
                                ] = endBarcodeObj.adapter
                                barcodeEntry[
                                    END_BARCODE_ANNOTATION
                                ] = endBarcodeObj.annotation
                                barcodeEntry[END_BARCODE_INDEX] = endBarcodeObj.index
                                barcodeEntry[
                                    END_BARCODE_SEQUENCE
                                ] = endBarcodeObj.sequence
                                barcodeEntry[END_BARCODE_TYPE] = endBarcodeObj.type
                                dualBarcodes_str = [str(x) for x in dualBarcodes]
                                matchingDualBarcodes = filter(
                                    lambda dualBarcode: dualBarcode.startswith(
                                        barcodeName
                                    ),
                                    dualBarcodes_str,
                                )
                                if len(matchingDualBarcodes) > 0:
                                    barcodeEntry[
                                        DUAL_BARCODE_NAME
                                    ] = matchingDualBarcodes[0]
                            else:
                                logger.error(
                                    ERROR_BARCODE_NOT_FOUND.format(
                                        barcodeName=barcodeName,
                                        barcodeKit=self.eas.barcodeKitName,
                                    )
                                )
                        else:
                            barcodeEntry[DUAL_BARCODE_NAME] = barcodeEntry[BARCODE_NAME]
                except Exception:
                    # intentionally do nothing....
                    pass

            referenceObj = get_reference(barcodeEntry[REFERENCE], tmap_version)
            barcodeEntry[REFERENCE_FULL_PATH] = (
                os.path.join(
                    referenceObj.reference_path, "%s.fasta" % barcodeEntry[REFERENCE]
                )
                if referenceObj
                else ""
            )
            barcodeEntry[REFERENCE_FULL_NAME] = (
                referenceObj.name if referenceObj else ""
            )
            barcodeEntry[ALIGNED] = (
                (REFERENCE in barcodeEntry)
                and bool(barcodeEntry[REFERENCE])
                and not barcodeEntry[FILTERED]
            )
            barcodeEntry[BAM] = dataset["file_prefix"] + (
                ".bam" if barcodeEntry[ALIGNED] else ".basecaller.bam"
            )
            barcodeEntry[BAM_FULL_PATH] = BarcodeSampleInfo.getFullBamPath(
                reportFullPath,
                barcodeEntry[REFERENCE],
                barcodeEntry[BAM],
                barcodeEntry[FILTERED],
            )

            # if a sample has been defined the sample id should be the primary key to look up the sample by
            if barcodeEntry[SAMPLE] != "":
                sample = self.result.experiment.samples.filter(
                    displayedName=barcodeEntry[SAMPLE]
                )
                barcodeEntry[SAMPLE_ID] = (
                    sample.first().externalId if sample.count() == 1 else ""
                )

            # construct the reference genome url which is web accessible
            barcodeEntry[GENOME_URL] = (
                "/auth/output/"
                + tmap_version
                + "/"
                + barcodeEntry[REFERENCE]
                + "/"
                + barcodeEntry[REFERENCE]
                + ".fasta"
                if barcodeEntry[REFERENCE]
                else ""
            )

            # assert the entry
            data[barcodeName] = barcodeEntry

        return data

    def data(self, start_json=None):
        """
        This will get a dictionary structure which represents keys for the names of the barcodes and values being a dictionary for each of the data points
        :param start_json:
        :return:
        """

        # we are going to handle unbarcoded data as a special case where as a single item is generated
        if not self.eas.barcodeKitName:
            return self.dataNonBarcoded()
        else:
            return self.dataBarcoded()

    @cached_property
    def barcodedSamples(self):
        barcodedSamples = self.eas.barcodedSamples
        if isinstance(barcodedSamples, basestring):

            try:
                barcodedSamples = json.loads(
                    barcodedSamples, encoding=settings.DEFAULT_CHARSET
                )
            except ValueError as j:
                try:
                    ## What is this? Was barcodedSamples getting python string dumps?
                    barcodedSamples = ast.literal_eval(barcodedSamples)
                except Exception as e:
                    logger.fatal(
                        "Unable to parse inputs as json [%s] or python [%s]", j, e
                    )
        try:
            for k, v in list(barcodedSamples.items()):
                if isinstance(v["barcodes"], list):
                    for bc in v["barcodes"]:
                        if not isinstance(bc, str):
                            logger.error("INVALID bc - NOT an str - bc=%s" % (bc))
                else:
                    logger.error(
                        "INVALID v[barcodes] - NOT a list!!! v[barcodes]=%s"
                        % (v["barcodes"])
                    )
        except Exception:
            logger.exception("Invalid barcodedSampleInfo value")
        return barcodedSamples

    @cached_property
    def datasetsBaseCaller(self):
        datasetsBaseCallerFile = os.path.join(
            self.result.get_report_path(),
            "basecaller_results",
            "datasets_basecaller.json",
        )
        with open(datasetsBaseCallerFile) as f:
            datasetsBaseCaller = json.load(f, encoding=settings.DEFAULT_CHARSET)

        # File is tagged with a version number (yay!). Best check it.
        ver = datasetsBaseCaller.get("meta", {}).get("format_version", "0")
        if ver != "1.0":
            logger.warn("Basecaller JSON file syntax has unrecognized version: %s", ver)
        return datasetsBaseCaller
