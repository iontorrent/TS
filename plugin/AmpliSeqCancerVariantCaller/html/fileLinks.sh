#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

print_html_file_links()
{
echo -n \
"             <div id=\"FileLinks\" class=\"report_block\">
                <h2>File Links</h2>                
				<br/>
				<div id=\"accordion\">

					<h3><a href=\"#\">Full Library Alignments (BAM)</a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/sorted.bam\">
					Click this link to download the alignment to hg19. 
					</a><br/><br/>
					The alignments are produced in the <a href=\"http://samtools.sourceforge.net/SAM1.pdf\">Sequence Alignment/Map</a> (SAM) format, but given as a BAM (binary SAM) file and can be directly loaded in IGV.
					</p></div>

					<h3><a href=\"#\">Library Alignments Index (BAM Index)</a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/sorted.bam.bai\">
                    Click this link to download the index for the alignments.  
					</a><br/><br/>
					The BAM file is indexed to allow retrieval of all reads aligned to a given locus.
					</p></div>
					
					<h3><a href=\"#\">Variant Calls (SNP)</a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/dibayes_out/SNP_variants.vcf.gz\">
                    Click this link to download the SNP calls. 
					</a><br/><br/>
					The variant calls are produced in the Variant Calling Format (VCF) and can be directly loaded in IGV
					</p></div>
					
					<h3><a href=\"#\"> Variant Calls Index (SNP) </a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/dibayes_out/SNP_variants.vcf.gz.tbi\">
                    Click this link to download the index for the variant calls (SNP). 
					</a><br/><br/>
					The VCF file is indexed by tabix to facilitate on-demand variant retrieval.
					</p></div>
					
					<h3><a href=\"#\">Variant Calls (Indel)</a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/indel_variants.vcf.gz\">
                    Click this link to download the Indel calls. 
					</a><br/><br/>
					The variant calls are produced in the Variant Calling Format (VCF) and can be directly loaded in IGV
					</p></div>
					
					<h3><a href=\"#\"> Variant Calls Index (Indel) </a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/indel_variants.vcf.gz.tbi\">
                    Click this link to download the index for the variant calls (Indel). 
					</a><br/><br/>
					The VCF file is indexed by tabix to facilitate on-demand variant retrieval.
					</p></div>
					
                                        <h3><a href=\"#\"> BED file for Amplicon Regions</a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/amplicon_seq.bed\">
                                        Click this link to download the BED file for the amplicon reqions. 
					</a><br/><br/>
					</p></div>
					
                                        <h3><a href=\"#\"> BED file for Cancer Panel Loci </a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/Cancer_panel_loci.bed\">
                                        Click this link to download the BED file for the Cancer panel loci.
					</a><br/><br/>
					</p></div>
                                        <h3><a href=\"#\"> Spreadsheet file with complete list of SNP calls </a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/dibayes_out/SNP_variants.xls\">
                                        Click this link to download the xls file for SNP variants.
					</a><br/><br/>
					</p></div>
                                        <h3><a href=\"#\"> Spreadsheet file with complete list of indel calls </a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/indel_variants.xls\">
                                        Click this link to download the xls file for indel variants.
					</a><br/><br/>
					</p></div>
                                        <h3><a href=\"#\"> Spreadsheet file with complete list of allele counts </a></h3>
					<div><p>
					<a href=\"${TSP_URLPATH_PLUGIN_DIR}/allele_counts.xls\">
                                        Click this link to download the xls file for allele counts.
					</a><br/><br/>
					</p></div>

                    
";

# Continue
echo -n \
"                   
            </div> <!--  end of files -->
";
}
