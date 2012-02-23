#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

print_html_file_links()
{
echo -n \
"             <div id=\"FileLinks\" class=\"report_block\">
                <h2>File Links</h2>                
				<br/>
				<div id=\"accordion\">
                    <h3><a href=\"#\">Browse Variants and Alignments in IGV</a></h3>
                    <div><p>
					<a id='igvLink'>
					Click this link to view the alignments and variant calls in the Integrated Genomics Viewer (IGV).
					</a><br/><br/>
					If the Reference Genome FASTA has not been previously imported into IGV, please import this FASTA using \"File\"->\"Import Genome\".
					It is suggested to use the name \"${TSP_LIBRARY}\" to identify this Reference Genome.
					This need only be done once for a given Reference Genome FASTA, regardless of the experiment or data.
					<br/><br/>
					This link has been tested with IGV version 2.0.3 - if you encounter problems, please download IGV with this version and run manually.
					</p></div>                    

					<h3><a href=\"#\">Full Library Alignments (BAM)</a></h3>
					<div><p>
					<a href=\"${PLUGIN_OUT_BAM_URL}\">
					Click this link to download the full library alignments. 
					</a><br/><br/>
					The alignments are produced in the <a href=\"http://samtools.sourceforge.net/SAM1.pdf\">Sequence Alignment/Map</a> (SAM) format, but given as a BAM (binary SAM) file and can be directly loaded in IGV.
					</p></div>

					<h3><a href=\"#\">Library Alignments Index (BAM Index)</a></h3>
					<div><p>
					<a href=\"${PLUGIN_OUT_BAM_URL}.bai\">
                    Click this link to download the index for the full library alignments.  
					</a><br/><br/>
					The BAM file is indexed to allow retrieval of all reads aligned to a given locus.
					</p></div>

                    <h3><a href=\"#\">Full Library Variant Calls (VCF)</a></h3>
                    <div><p>
					<a href=\"${PLUGIN_OUT_VCF_GZ_VARIANTS}\">
					Click this link to download the variant calls.  
					</a><br/><br/>
					The variant calls are produced in the <a href=\"http://www.1000genomes.org/wiki/Analysis/Variant%20Call%20Format/vcf-variant-call-format-version-41\">Variant Calling Format</a> (VCF) and can be directly loaded in IGV.
					</p></div>

                    <h3><a href=\"#\">Library Variant Calls Index (TBI)</a></h3>
                    <div><p>
					<a href=\"${PLUGIN_OUT_VCF_GZ_VARIANTS}.tbi\">
					Click this link to download the index for the variant calls.  
					</a><br/><br/>
					The VCF file is indexed by <a href=\"http://samtools.sourceforge.net/tabix.shtml\">tabix</a> to facilitate on-demand variant retrieval.
					</p></div>
";
# Output the extra links
if test 1 = ${ALL_VARIANTS}; then
    echo -e "\t\t\t\t\t\t\t<h3><a href=\"#\">Full Library Calls (VCF)</a></h3>";
    echo -e "\t\t\t\t\t\t\t<div><p><a href="${PLUGIN_OUT_VCF_GZ_ALL}">";
	echo -e "\t\t\t\t\t\t\tClick this link to download the calls for all positions."
	echo -e "\t\t\t\t\t\t\t</a><br/><br/>";
	echo -e "\t\t\t\t\t\t\tThe variant calls are produced in the Variant Calling Format (VCF).";
	echo -e "\t\t\t\t\t\t\t</p></div>";
	echo -e "\t\t\t\t\t\t\t<h3><a href=\"#\">Full Library Calls Index (TBI)</a></h3>";
    echo -e "\t\t\t\t\t\t\t<div><p><a href="${PLUGIN_OUT_VCF_GZ_ALL}.tbi">";
	echo -e "\t\t\t\t\t\t\tClick this link to download the index for all positions."
	echo -e "\t\t\t\t\t\t\t</a><br/><br/>";
	echo -e "\t\t\t\t\t\t\tThe VCF file is indexed by <a href=\"http://samtools.sourceforge.net/tabix.shtml\">tabix</a> to facilitate on-demand variant retrieval.";
	echo -e "\t\t\t\t\t\t\t</p></div>";
fi
# Continue
echo -n \
"                   <h3><a href=\"#\">Reference Genome (FASTA)</a></h3>
                    <div><p>
					<a href=\"${TSP_URLPATH_GENOME_FASTA}\">
                    Click this link to download the reference genome FASTA.
					</a>
					</p></div>
                    <h3><a href=\"#\">Detailed Log (TXT)</a></h3>
                    <div><p>
					<a href=\"${PLUGIN_OUT_DETAILED_LOG}\">
                    Click this link to download the detailed log from this plugin.
					</a>
					</p></div>
                </div>
            </div> <!--  end of files -->
";
}
