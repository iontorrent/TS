#!/usr/bin/python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

''' Temporary runner for testing basecalling and post_basecalling'''
if __name__=="__main__":
    
    env = {
        'SIGPROC_RESULTS'       : '../sigproc_results',
        'basecallerArgs'        : '/home/msikora/Documents/BaseCaller',
        'libraryKey'            : 'TCAG',
        'tfKey'                 : 'ATCG',
        'runID'                 : 'ABCDE',
        'flowOrder'             : 'TACGTACGTCTGAGCATCGATCGATGTACAGC',
        'reverse_primer_dict'   : {'adapter_cutoff':16,'sequence':'ATCACCGACTGCCCATAGAGAGGCTGAGAC','qual_window':30,'qual_cutoff':9},
        'BASECALLER_RESULTS'    : 'basecaller_results',
        'barcodeId'             : 'IonExpress',
        #'barcodeId'             : '',
        'referenceName'         : 'hg19',
        'sample'                : 'My-sample',
        'site_name'             : 'My-site',
        'notes'                 : 'My-notes',
        'start_time'            : time.asctime(),
        'align_full'            : False,
        'flows'                 : 260,
        'aligner_opts_extra'    : '',
        'mark_duplicates'       : False,
        'ALIGNMENT_RESULTS'     : './',
        'sam_parsed'            : False,
        'chipType'              : '316B',
        'expName'               : 'My-experiment',
        'resultsName'           : 'My-results',
        'instrumentName'        : 'B19'
    }

    
    basecalling(
        env['SIGPROC_RESULTS'],
        env['basecallerArgs'],
        env['libraryKey'],
        env['tfKey'],
        env['runID'],
        env['flowOrder'],
        env['reverse_primer_dict'],
        env['BASECALLER_RESULTS'],
        env['barcodeId'],
        env.get('barcodeSamples',''),
        os.path.join("barcodeList.txt"),
        os.path.join(env['BASECALLER_RESULTS'], "barcodeMask.bin"),
        env['referenceName'],
        env['sample'],
        env['site_name'],
        env['notes'],
        env['start_time'],
        env['chipType'],
        env['expName'],
        env['resultsName'],
        env['instrumentName']
    )
    
    post_basecalling(env['BASECALLER_RESULTS'],env['expName'],env['resultsName'],env['flows'])

    
    tf_processing(
        os.path.join(env['BASECALLER_RESULTS'], "rawtf.basecaller.bam"),
        env['tfKey'],
        env['flowOrder'],
        env['BASECALLER_RESULTS'],
        '.')

    
    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed')):
        post_basecalling(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.untrimmed'),env['expName'],env['resultsName'],env['flows'])
    
    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed')):
        post_basecalling(os.path.join(env['BASECALLER_RESULTS'],'unfiltered.trimmed'),env['expName'],env['resultsName'],env['flows'])
    

    
    #from ion.utils import alignment
    import alignment
    bidirectional = False
    

    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed")):
        alignment.alignment_unmapped_bam(
            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed"),
            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.untrimmed"),
            env['align_full'],
            env['referenceName'],
            env['flows'],
            env['aligner_opts_extra'],
            env['mark_duplicates'],
            bidirectional,
            env['sam_parsed'])

    if os.path.exists(os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed")):
        alignment.alignment_unmapped_bam(
            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed"),
            os.path.join(env['BASECALLER_RESULTS'],"unfiltered.trimmed"),
            env['align_full'],
            env['referenceName'],
            env['flows'],
            env['aligner_opts_extra'],
            env['mark_duplicates'],
            bidirectional,
            env['sam_parsed'])

    alignment.alignment_unmapped_bam(
        env['BASECALLER_RESULTS'],
        env['ALIGNMENT_RESULTS'],
        env['align_full'],
        env['referenceName'],
        env['flows'],
        env['aligner_opts_extra'],
        env['mark_duplicates'],
        bidirectional,
        env['sam_parsed'])
