#!/bin/bash
# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved

print_html_about()
{
echo \
'            <div id="About" class="report_block">                
                <a name="about"><h2>About</h2></a>
                <div style>
					</br>
					<b>Please follow the links below for detailed information on the tools and algorithms used to generate these results.</b>
					</br>
					<ul>
						<li>
							The Germ-line Variant Caller uses <a href="http://samtools.sourceforge.net">SAMTools/BCFtools</a> to identify SNPs/INDELs. 
							<ul>
								<li>
									The following command line was used to call variants:
									<br/>
									<span style="font-family: courier new,courier,monospace; font-size: 12px">
'
echo -n \
"										samtools mpileup -d ${SAMTOOLS_MAX_DEPTH} -L ${SAMTOOLS_MAX_DEPTH_INDEL} -Q ${SAMTOOLS_MIN_BASEQ} -h ${SAMTOOLS_HP_COEFF} -o ${SAMTOOLS_PHRED_GAPO} -e ${SAMTOOLS_PHRED_GAPE} -m ${SAMTOOLS_MIN_GAPPED_READS} -f [reference.fasta] -g [run.bam] | \\
										<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
										bcftools view -e -g -c -v -N -i ${BCFTOOLS_INDEL_TO_SUB_RATIO} - | \\
										<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
										perl vcf_filter.pl -d ${VCF_FILTER_MIN_READ_DEPTH} -D ${VCF_FILTER_MAX_READ_DEPTH} -w ${VCF_FILTER_WIN_SNP_ADJ_GAPS} -W ${VCF_FILTER_WIN_INDEL_ADJ_GAPS} -s ${VCF_FILTER_SNPS_STRAND_DEPTH} -S ${VCF_FILTER_INDELS_STRAND_DEPTH} -H ${VCF_FILTER_HPS_STRAND_DEPTH} | \\
										<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
										bgzip &gt; [variants.vcf.gz]
";
echo \
'									</span>
									<br/>
									<span style="font-family: courier new,courier,monospace; font-size: 12px">
										tabix -p vcf [variants.vcf.gz]
									</span>
								</li>
								<li>
									The exact parameters used to generate the variant list can be found in the Detailed Log file located in the <a href="#FileLinks">File Links</a> section on the variant calling page.  
								</li>
							</ul>
						</li>
						<li>
							The alignment are provided in the <a href="http://samtools.sourceforge.net/SAM1.pdf">Sequence Alignment/Map</a> (SAM) format.
						</li>
						<li>
							The final variant calls are provided in the standard <a href="http://www.1000genomes.org/wiki/Analysis/Variant%20Call%20Format/vcf-variant-call-format-version-41">Variant Calling Format</a> (VCF).
							<ul>
								<li>
									Phred-scaled quality score (Q) are calculated using -10*log10 Pr(no variant).
								</li>
								<li>
									High quality scores indicate high confidence calls.
								</li>
							</ul>
						</li>
						<li>
							These standard community formats were created by the <a href="http://www.1000genomes.org">1000 Genomes Project</a>.
						</li>
						<li>
							The genome browser linked here is the <a href="http://www.broadinstitute.org/software/igv/">Integrated Genomics Viewer</a> or IGV. 
							<ul>
								<li>
									Please consult the documentation on their website for more information on how to use IGV or for troubleshooting.
								</li>
								<li>
									IGV requires Java version 6, try running the <a href="http://www.javatester.org/version.html">Java Tester</a> if you do not know your current Java version.
								</li>
								<li>
									Please remember to import the Reference Genome FASTA before viewing data aligned to the FASTA for the first time.
								</li>
							</ul>
						</li>
					</ul>
                </div>
            </div> <!--  end of about-->';
}
