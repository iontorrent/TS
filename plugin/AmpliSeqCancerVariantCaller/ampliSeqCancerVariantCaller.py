# Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved
import json
import sys
import os
#import samutils
import diBayes
import json_utils
import alignment_qc
import barcodeutils
import subprocess


JSON_INPUT = ""
DEV_STDIN = "/dev/stdin"
DEV_STDERR = "/dev/stderr"
DEV_STDOUT = "/dev/stdout"
DEF_PICARD_ARGS = "VALIDATION_STRINGENCY=SILENT"
GATK="/results/plugins/AmpliSeqCancerVariantCaller/GATK/dist/GenomeAnalysisTK.jar"
BGZIP="/results/plugins/Germ-lineVariantCaller/bgzip"
TABIX="/results/plugins/Germ-lineVariantCaller/tabix"
INPUT_BED_FILE="/results/plugins/AmpliSeqCancerVariantCaller/bedfiles/400_hsm_v12_1_seq.bed"
GATK_VARIANTS="/variants.vcf"
INPUT_SNP_BED_FILE="/results/plugins/AmpliSeqCancerVariantCaller/bedfiles/HSM_ver12_1_loci.bed"


def IndelCallerCommandLine(jsonParams, bam, reference, outDir):                                         
    binDir = jsonParams['runinfo']['plugin_dir']
    commandLine = """ %s/ion-variant-hunter-core  --dynamic-space-size 6144 --function write-alignment-n-deviation-lists-from-sam --bam-file %s --reference-file %s --base-output-filename %s/%s --fs-align-jar-args 'VALIDATION_STRINGENCY=SILENT' --java-bin `which java`  --java-bin-args _Xmx4G --strand-prob ,0 """ % (binDir, bam, reference, outDir, "variantCalls")
    return commandLine


def RunInParallel(commands):
    
    processCounter = 0
    processes = set()
    #print "commands: ", commands
    my_env = os.environ
    my_env['LOG4CXX_CONFIGURATION'] = JSON_INPUT['runinfo']['plugin_dir'] + "/log4j.properties"
    for cmd in commands:
        cmd = cmd.strip()
        processName = "%s-%d" % ("variant-process", processCounter)
        print "%s" % (cmd)
        processes.add( subprocess.Popen(cmd, shell=True, env=my_env) )
        processCounter += 1
        #processes.difference_update( p for p in processes if p.poll() is not None)
        
    for p in processes:
        if p.poll() is not None:
            p.wait()
    
def RunCommand(command):
    command = command.strip()
    print "[ampliSeqCancerVariantCaller] executing: ", command

    os.system( command )

def FilterSFFListByBarcodeName( barcodes, sffs, outputDir ):
    """filters sffs by barcode id"""
    #make a simple dict of barcode id, sff
    barcodeDict = {}
    for barcode in barcodes:
        barcodeDict[ barcode.idString ] = True
    #t = [ sff for sff in sffs for idString in barcodeDict if sff.find(idString) != -1 ]
    returnDict = {} #maps barcodeId to tuples of (sff, bam) file names
    
    for sff in sffs:
        for idString in barcodeDict:
            if sff.find( idString ) != -1:
                bam = outputDir + "/" +idString+"/"+ idString + "_sorted.bam"
                flowbam = outputDir + "/" +idString+"/"+ idString + "_sorted_flowspace.bam"
                #this is awful.  once the plugin directory structure changes this is broken
                if sff.find("..") == -1:
                    sff = outputDir + "/../../" + sff 
                returnDict[ idString ] = ( sff, bam, flowbam )
    return returnDict

def RunTMAPOnList( jarLocation, sffsandbams, reference ):
    #sort bam
     #realign reads without flowspace information
    for barcodeId in sffsandbams:
        sff, bam, flowbam = sffsandbams[barcodeId]
        if not os.path.exists(os.path.dirname(bam)):
            os.makedirs(os.path.dirname(bam))
        convertToBAM = "java -Xmx8G -jar %s/SamFormatConverter.jar I=%s O=%s %s" % ( jarLocation, DEV_STDIN, DEV_STDOUT, DEF_PICARD_ARGS )
        sortBAM = "java -Xmx8G -jar %s/SortSam.jar I=%s O=%s SO=coordinate %s" % (jarLocation, DEV_STDIN, bam, DEF_PICARD_ARGS)
        pipedConvertAndSortCmd = "%s | %s" % (convertToBAM, sortBAM)

        tmap = alignment_qc.TMAP()
        tmap.SetReference( reference  )
        tmap.SetReadsSFF( sff )
        tmap.AddParam("R","\"ID:AmpliSeq\"")
        tmap.AddParam("R","\"PU:PGM\"")
        tmap.AddParam("R","\"SM:hg19\"")
        tmap.AddParam("R","\"LB:hg19\"")
        tmap.AddParam("R","\"PL:IONTORRENT\"")
        tmap.SetOutputPipeDestination( pipedConvertAndSortCmd )
        tmap.Align()
         #index non-flowspace bam
        bamIndex = "%s.bai" % ( bam )
        indexBAM = "java -Xmx8G -jar %s/BuildBamIndex.jar I=%s O=%s %s" % ( jarLocation, bam, bamIndex, DEF_PICARD_ARGS )
        RunCommand( indexBAM )

        #realign reads with flowspace information                                               
        convertToBAM = "java -Xmx8G -jar %s/SamFormatConverter.jar I=%s O=%s %s" % ( jarLocation, DEV_STDIN, DEV_STDOUT, DEF_PICARD_ARGS )
        sortBAM = "java -Xmx8G -jar %s/SortSam.jar I=%s O=%s SO=coordinate %s" % (jarLocation, DEV_STDIN, flowbam, DEF_PICARD_ARGS)
        pipedConvertAndSortCmd = "%s | %s" % (convertToBAM, sortBAM)
        tmap = alignment_qc.TMAP()                                                              
        tmap.SetReference(reference )                               
        tmap.SetReadsSFF( sff )
        tmap.AddParam("Y")                                                                      
        tmap.AddParam("R","\"ID:AmpliSeq\"")                                                    
        tmap.AddParam("R","\"PU:PGM\"")                                                         
        tmap.AddParam("R","\"SM:hg19\"")                                                        
        tmap.AddParam("R","\"LB:hg19\"")                                                        
        tmap.AddParam("R","\"PL:IONTORRENT\"")                                                  
        tmap.SetOutputPipeDestination( pipedConvertAndSortCmd )                                 
        tmap.Align()         

        bamIndex = "%s.bai" % ( flowbam )
        indexBAM = "java -Xmx8G -jar %s/BuildBamIndex.jar I=%s O=%s %s" % ( jarLocation, flowbam, bamIndex, DEF_PICARD_ARGS )
        RunCommand( indexBAM )


    

if __name__ == '__main__':
    #JSON_INPUT = json.load( open(sys.argv[1], "r") )
    JSON_INPUT = json.load( open(sys.argv[1], "r") )
    PLUGIN_RESULTS_DIR = JSON_INPUT['runinfo']['results_dir']
    JARS = JSON_INPUT['runinfo']['plugin_dir']

    analysisDir = json_utils.AnalysisDir( JSON_INPUT['runinfo']['analysis_dir'] )
    sffs = []
    sffsandbams = {}

    if analysisDir.barcodeList: #run is barcoded
        files = os.listdir(str(analysisDir))
        sffs = [ sff for sff in files if sff.find(".sff") != -1 ]
        sffsandbams = FilterSFFListByBarcodeName( analysisDir.barcodeList.barcodes, sffs, PLUGIN_RESULTS_DIR )
    else: #run is not barcoded
        #str(dibayes.GetAnalysisDir()) +"/"+dibayes.GetAnalysisDir().GetSFF()
        sff = JSON_INPUT['runinfo']['analysis_dir']+"/"+analysisDir.GetSFF()
        bam = PLUGIN_RESULTS_DIR + "/sorted.bam"
        flowbam = PLUGIN_RESULTS_DIR + "/sorted_flowspace.bam"
        sffsandbams = {}
        sffsandbams["/"] = ( sff, bam, flowbam )
    
    
    #init SNP caller    
    dibayes = diBayes.DiBayes( JSON_INPUT['runinfo']['results_dir'] )
    dibayes.Init("hg19")
    
    #GATK SETUP
    #run "java -Xmx8g -jar ${GATK} -T UnifiedGenotyper -R /results/referenceLibrary/tmap-f2/hg19/hg19.fasta -stand_call_conf 40.0 -stand_emit_conf 30.0 -L ${INPUT_SNP_BED_FILE} -dcov 8000 -nt 4 -I ${TSP_FILEPATH_PLUGIN_DIR}/sorted.bam -o ${TSP_FILEPATH_PLUGIN_DIR}/${PLUGIN_OUT_GATK_VARIANTS} -glm INDEL -minIndelCnt 10";
    RunTMAPOnList( JARS, sffsandbams, dibayes.referenceLibrary.FastaPath() )
    for runtype in sffsandbams:
        snp_dir = PLUGIN_RESULTS_DIR+"/"+runtype+"/dibayes_out/"
        indel_dir = PLUGIN_RESULTS_DIR+"/"+runtype
        if not os.path.exists(snp_dir):
            os.makedirs(snp_dir)
        if not os.path.exists(indel_dir):
            os.makedirs(indel_dir)
        sff, bam, flowbam = sffsandbams[ runtype ]
        GATK_REF="/results/referenceLibrary/tmap-f2/hg19/hg19.fasta"
        GATK_CMD="java -Xmx8g -jar %s -T UnifiedGenotyper -R %s -stand_call_conf 40.0 -stand_emit_conf 30.0 -L %s -dcov 8000 -nt 4 -I %s -o %s/%s -glm INDEL -minIndelCnt 10" % ( GATK, GATK_REF, INPUT_SNP_BED_FILE, bam, PLUGIN_RESULTS_DIR, runtype +"/" + GATK_VARIANTS )

        dibayes.bamFile = bam
        dibayes.out_dir = PLUGIN_RESULTS_DIR  + "/" + runtype + "/dibayes_out/"
        dibayes.log_dir = PLUGIN_RESULTS_DIR + "/" + runtype + "/dibayes_out/log"
        indelCommand = IndelCallerCommandLine( JSON_INPUT, flowbam, dibayes.referenceLibrary.FastaPath(), indel_dir )
        
        RunCommand("export LOG4CXX_CONFIGURATION=/results/plugins/AmpliSeqCancerVariantCaller/log4j.properties;"+dibayes.CommandLine())
        #RunCommand(GATK_CMD)
        RunCommand(indelCommand)
